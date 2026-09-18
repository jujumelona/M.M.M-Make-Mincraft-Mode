package mmm.owner;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.*;
import org.eclipse.core.resources.*;
import org.eclipse.core.runtime.*;
import org.eclipse.equinox.app.IApplication;
import org.eclipse.equinox.app.IApplicationContext;
import org.eclipse.jdt.core.*;
import org.eclipse.jdt.apt.core.util.AptConfig;
import org.eclipse.jdt.apt.core.util.IFactoryPath;

/** Single-writer, persistent Eclipse resource workspace. stdout is exclusively JSON lines. */
public final class OwnerApplication implements IApplication {
    private final Gson gson = new GsonBuilder().serializeNulls().create();
    private final String session = UUID.randomUUID().toString();
    private final Map<String, String> projectIds = new LinkedHashMap<>();
    private IWorkspace workspace;
    private long generation;
    private boolean opened;
    private boolean closing;

    private static void stage(String value) {
        System.err.println("MMM_OWNER_STAGE " + value);
        System.err.flush();
    }

    @Override public Object start(IApplicationContext context) throws Exception {
        PrintStream protocol = System.out;
        System.setOut(System.err);
        stage("application.start");
        workspace = ResourcesPlugin.getWorkspace();
        stage("application.workspace.ready");
        IWorkspaceDescription desc = workspace.getDescription();
        desc.setAutoBuilding(false);
        workspace.setDescription(desc);
        stage("application.input_loop.ready");
        try (BufferedReader input = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            String line;
            while (!closing && (line = input.readLine()) != null) {
                stage("application.request.received");
                Object id = null;
                Map<String, Object> reply = new LinkedHashMap<>();
                try {
                    Map<String, Object> request = gson.fromJson(line, new TypeToken<Map<String, Object>>(){}.getType());
                    id = request.get("id");
                    @SuppressWarnings("unchecked") Map<String, Object> params = (Map<String,Object>)request.getOrDefault("params", Map.of());
                    Object result = switch (String.valueOf(request.get("method"))) {
                        case "resolve" -> GradleModels.resolve(params);
                        case "open" -> open(params);
                        case "build" -> build(Boolean.TRUE.equals(params.get("full")));
                        case "diagnostics" -> { requireOpen(); yield snapshot(); }
                        case "close" -> { closing = true; workspace.save(true, null); yield Map.of("closed", true); }
                        default -> throw new IllegalArgumentException("Unknown owner method");
                    };
                    reply.put("result", result);
                } catch (Exception failure) {
                    failure.printStackTrace(System.err);
                    reply.put("error", Map.of("message", failure.toString()));
                }
                reply.put("id", id);
                protocol.println(gson.toJson(reply)); protocol.flush();
            }
        } finally { workspace.save(true, null); }
        return IApplication.EXIT_OK;
    }

    @SuppressWarnings("unchecked")
    private Object open(Map<String, Object> params) throws Exception {
        stage("open.begin");
        Map<String, Object> model = (Map<String,Object>)params.get("model");
        if (model == null) throw new IllegalArgumentException("model is required");
        List<Map<String,Object>> sets = (List<Map<String,Object>>)model.get("source_sets");
        if (sets == null || sets.isEmpty()) throw new IllegalArgumentException("No resolved Java source sets");
        // Reopening is a complete model replacement, never reuse previous marker/classpath state.
        opened = false;
        for (IProject project : workspace.getRoot().getProjects()) project.delete(true, true, null);
        stage("open.previous_projects_deleted");
        projectIds.clear();
        Map<String,IJavaProject> projects = new LinkedHashMap<>();
        Map<String,String> outputs = new HashMap<>();
        for (Map<String,Object> set : sets) {
            String id = required(set, "id");
            stage("open.project.create.begin:" + id);
            String name = "source_" + UUID.nameUUIDFromBytes(id.getBytes(StandardCharsets.UTF_8)).toString().replace("-", "");
            IProject project = workspace.getRoot().getProject(name);
            project.create(null); project.open(null);
            IProjectDescription description = project.getDescription();
            description.setNatureIds(new String[]{JavaCore.NATURE_ID});
            ICommand command = description.newCommand(); command.setBuilderName(JavaCore.BUILDER_ID);
            description.setBuildSpec(new ICommand[]{command}); project.setDescription(description, null);
            IJavaProject javaProject = JavaCore.create(project);
            projects.put(id, javaProject); projectIds.put(name, id);
            for (String output : strings(set, "output_dirs")) outputs.put(canonical(output), id);
            for (String output : strings(set, "output_artifacts")) outputs.put(canonical(output), id);
            stage("open.project.create.end:" + id);
        }
        for (Map<String,Object> set : sets) {
            String id = required(set, "id");
            stage("open.configure.begin:" + id);
            configure(set, projects, outputs);
            stage("open.configure.end:" + id);
        }
        opened = true;
        stage("open.full_build.begin");
        Object result = build(true);
        stage("open.full_build.end");
        return result;
    }

    private void configure(Map<String,Object> set, Map<String,IJavaProject> projects, Map<String,String> outputs) throws Exception {
        IJavaProject javaProject = projects.get(required(set, "id"));
        IProject project = javaProject.getProject();
        List<String> processors = strings(set, "annotation_processor_path");
        List<String> args = strings(set, "compiler_args");
        Map<String,String> processorOptions = new LinkedHashMap<>();
        for (String arg : args) {
            if (arg.startsWith("-A") && arg.length() > 2) {
                String[] option = arg.substring(2).split("=", 2);
                processorOptions.put(option[0], option.length == 2 ? option[1] : "");
            } else if (!arg.equals("-parameters") && !arg.equals("-proc:none") && !arg.equals("-nowarn") && !arg.startsWith("-Xlint")) {
                throw new IllegalArgumentException("Unsupported compiler argument: " + arg);
            }
        }
        List<IClasspathEntry> entries = new ArrayList<>();
        String javaHome = canonical(required(set, "java_home"));
        java.nio.file.Path jrt = java.nio.file.Path.of(javaHome, "lib", "jrt-fs.jar");
        if (Files.isRegularFile(jrt)) {
            entries.add(JavaCore.newLibraryEntry(new Path(jrt.toString()), null, null, new IAccessRule[0],
                new IClasspathAttribute[]{JavaCore.newClasspathAttribute(IClasspathAttribute.MODULE, "true")}, false));
        } else {
            java.nio.file.Path rt = java.nio.file.Path.of(javaHome, "jre", "lib", "rt.jar");
            if (!Files.isRegularFile(rt)) throw new IllegalArgumentException("JDK runtime libraries are missing: " + javaHome);
            entries.add(JavaCore.newLibraryEntry(new Path(rt.toString()), null, null));
        }
        int index = 0;
        for (String source : strings(set, "source_roots")) {
            IFolder folder = project.getFolder("src" + index++);
            folder.createLink(new java.io.File(source).toURI(), IResource.ALLOW_MISSING_LOCAL, null);
            IPath[] includes = strings(set, "source_includes").stream().map(Path::new).toArray(IPath[]::new);
            IPath[] excludes = strings(set, "source_excludes").stream().map(Path::new).toArray(IPath[]::new);
            entries.add(JavaCore.newSourceEntry(folder.getFullPath(), includes, excludes, null));
        }
        Set<String> references = new LinkedHashSet<>();
        for (String library : strings(set, "classpath")) {
            String dependency = outputs.get(canonical(library));
            if (dependency != null) {
                if (!dependency.equals(required(set, "id")) && references.add(dependency))
                    entries.add(JavaCore.newProjectEntry(projects.get(dependency).getPath(), false));
            } else {
                if (!new File(library).exists()) throw new IllegalArgumentException("Resolved classpath does not exist: " + library);
                entries.add(JavaCore.newLibraryEntry(new Path(canonical(library)), null, null));
            }
        }
        IProjectDescription description = project.getDescription();
        description.setReferencedProjects(references.stream().map(id -> projects.get(id).getProject()).toArray(IProject[]::new));
        project.setDescription(description, null);
        IFolder bin = project.getFolder("bin"); bin.create(true, true, null);
        stage("configure.classpath.begin:" + required(set, "id"));
        javaProject.setRawClasspath(entries.toArray(IClasspathEntry[]::new), bin.getFullPath(), null);
        stage("configure.classpath.end:" + required(set, "id"));
        Map<String,String> options = new HashMap<>();
        String release = set.get("release") instanceof Number n ? Integer.toString(n.intValue()) : null;
        String source = release != null ? release : required(set, "source_compatibility");
        String target = release != null ? release : required(set, "target_compatibility");
        if (!JavaCore.getAllVersions().contains(source)) throw new IllegalArgumentException("Unsupported Java version: " + source);
        JavaCore.setComplianceOptions(source, options);
        options.put(JavaCore.COMPILER_SOURCE, source);
        options.put(JavaCore.COMPILER_CODEGEN_TARGET_PLATFORM, target);
        if (release != null) options.put(JavaCore.COMPILER_RELEASE, JavaCore.ENABLED);
        if (args.contains("-parameters")) options.put(JavaCore.COMPILER_CODEGEN_METHOD_PARAMETERS_ATTR, JavaCore.GENERATE);
        javaProject.setOptions(options);
        stage("configure.options.end:" + required(set, "id"));
        if (!processors.isEmpty() && !args.contains("-proc:none")) {
            IFactoryPath factoryPath = AptConfig.getDefaultFactoryPath(javaProject);
            List<String> reversed = new ArrayList<>(processors);
            Collections.reverse(reversed);
            for (String processor : reversed) {
                File jar = new File(processor);
                if (!jar.isFile()) throw new IllegalArgumentException("Processor must be a resolved JAR: " + processor);
                factoryPath.addExternalJar(jar);
            }
            AptConfig.setFactoryPath(javaProject, factoryPath);
            AptConfig.setProcessorOptions(processorOptions, javaProject);
            AptConfig.setGenSrcDir(javaProject, ".apt_generated");
            AptConfig.setEnabled(javaProject, true);
        }
    }

    private Object build(boolean full) throws Exception {
        requireOpen();
        String mode = full ? "full" : "incremental";
        // Eclipse compares resources and creates deltas; catches external writers as well as declared changes.
        stage("build." + mode + ".refresh.begin");
        workspace.getRoot().refreshLocal(IResource.DEPTH_INFINITE, null);
        stage("build." + mode + ".refresh.end");
        stage("build." + mode + ".workspace.begin");
        workspace.build(full ? IncrementalProjectBuilder.FULL_BUILD : IncrementalProjectBuilder.INCREMENTAL_BUILD, null);
        stage("build." + mode + ".workspace.end");
        generation++;
        stage("build." + mode + ".snapshot.begin");
        Object result = snapshot();
        stage("build." + mode + ".snapshot.end");
        return result;
    }

    private Map<String,Object> snapshot() throws CoreException {
        List<Map<String,Object>> diagnostics = new ArrayList<>();
        for (IMarker marker : workspace.getRoot().findMarkers(IMarker.PROBLEM, true, IResource.DEPTH_INFINITE)) {
            IResource resource = marker.getResource();
            Map<String,Object> row = new LinkedHashMap<>();
            row.put("path", resource.getLocation() == null ? "" : resource.getLocation().toOSString());
            row.put("uri", resource.getLocationURI() == null ? "" : resource.getLocationURI().toString());
            row.put("source_set", projectIds.get(resource.getProject().getName()));
            row.put("message", marker.getAttribute(IMarker.MESSAGE, ""));
            int severity = marker.getAttribute(IMarker.SEVERITY, IMarker.SEVERITY_ERROR);
            row.put("severity", severity == IMarker.SEVERITY_ERROR ? "error" : severity == IMarker.SEVERITY_WARNING ? "warning" : "info");
            row.put("line", marker.getAttribute(IMarker.LINE_NUMBER, 0));
            row.put("start", marker.getAttribute(IMarker.CHAR_START, -1));
            row.put("end", marker.getAttribute(IMarker.CHAR_END, -1));
            row.put("code", marker.getAttribute(IJavaModelMarker.ID, 0));
            row.put("marker_type", marker.getType());
            diagnostics.add(row);
        }
        return Map.of("session_id", session, "generation", generation, "diagnostics", diagnostics, "complete", true);
    }

    private void requireOpen() { if (!opened) throw new IllegalStateException("No initialized workspace model"); }
    private static String canonical(String path) throws IOException { return new File(path).getCanonicalPath(); }
    private static String required(Map<String,Object> map, String key) {
        Object value = map.get(key);
        if (value == null || value.toString().isBlank()) throw new IllegalArgumentException("Missing " + key);
        return value.toString();
    }
    private static List<String> strings(Map<String,Object> map, String key) {
        Object value = map.get(key);
        if (value == null) return List.of();
        if (!(value instanceof List<?> values)) throw new IllegalArgumentException("Expected list: " + key);
        return values.stream().map(Object::toString).toList();
    }
    @Override public void stop() { closing = true; }
}
