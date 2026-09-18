package mmm.owner;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;
import java.io.InputStream;
import java.io.Serializable;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.jar.JarEntry;
import java.util.jar.JarOutputStream;
import org.gradle.tooling.BuildAction;
import org.gradle.tooling.BuildActionExecuter;
import org.gradle.tooling.BuildController;
import org.gradle.tooling.GradleConnector;
import org.gradle.tooling.ProjectConnection;
import org.gradle.tooling.model.gradle.BasicGradleProject;
import org.gradle.tooling.model.gradle.GradleBuild;

/** Resolves effective JavaCompile inputs through a registered daemon-side model. */
public final class GradleModels {
    private GradleModels() {}

    public static Map<String, Object> resolve(Map<String, Object> params) throws Exception {
        Object rootValue = params.get("project_root");
        if (!(rootValue instanceof String) || ((String) rootValue).isBlank()) {
            throw new IllegalArgumentException("project_root is required");
        }
        Path root = Path.of((String) rootValue).toRealPath();
        Path init = Files.createTempFile("mmm-owner-model-", ".gradle");
        Path modelJar = Files.createTempFile("mmm-owner-model-api-", ".jar");
        try {
            // A minimal API JAR works with Equinox bundle URLs and avoids putting
            // the full owner/runtime or mutable workspace on Gradle's classpath.
            try (InputStream api = GradleSourceModel.class.getResourceAsStream("GradleSourceModel.class");
                 JarOutputStream jar = new JarOutputStream(Files.newOutputStream(modelJar))) {
                if (api == null) throw new IllegalStateException("Missing Gradle model interface bytecode");
                jar.putNextEntry(new JarEntry("mmm/owner/GradleSourceModel.class"));
                api.transferTo(jar);
                jar.closeEntry();
            }
            String prefix = "initscript { dependencies { classpath files('"
                    + groovyQuote(modelJar.toString()) + "') } }\n";
            Files.writeString(init, prefix + INIT_SCRIPT, StandardCharsets.UTF_8);
            GradleConnector connector = GradleConnector.newConnector().forProjectDirectory(root.toFile());
            Object gradleHome = params.get("gradle_home");
            if (gradleHome instanceof String && !((String) gradleHome).isBlank()) {
                connector.useInstallation(Path.of((String) gradleHome).toRealPath().toFile());
            }
            Object gradleUserHome = params.get("gradle_user_home");
            if (gradleUserHome instanceof String && !((String) gradleUserHome).isBlank()) {
                Path userHome = Path.of((String) gradleUserHome).toAbsolutePath().normalize();
                Files.createDirectories(userHome);
                connector.useGradleUserHomeDir(userHome.toFile());
            }
            try (ProjectConnection connection = connector.connect()) {
                BuildActionExecuter<List<String>> execution = connection.action(new ResolveAction());
                execution.withArguments("--init-script", init.toString(), "--console=plain");
                Object javaHome = params.get("java_home");
                if (javaHome instanceof String && !((String) javaHome).isBlank()) {
                    execution.setJavaHome(Path.of((String) javaHome).toRealPath().toFile());
                }
                // stdout belongs exclusively to the owner's RPC transport.
                execution.setStandardOutput(System.err);
                execution.setStandardError(System.err);
                List<String> models = execution.run();
                List<Map<String, Object>> sourceSets = new ArrayList<>();
                Map<String, Object> outputs = new LinkedHashMap<>();
                Map<String, Object> mappingModels = new LinkedHashMap<>();
                Gson gson = new Gson();
                String gradleVersion = null;
                for (String json : models) {
                    Map<String, Object> project = gson.fromJson(json,
                            new TypeToken<Map<String, Object>>() {}.getType());
                    if (gradleVersion == null) gradleVersion = (String) project.get("gradle_version");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> mapping = (Map<String, Object>) project.get("mapping_model");
                    if (mapping != null) mappingModels.put((String) project.get("project_dir"), mapping);
                    @SuppressWarnings("unchecked")
                    List<Map<String, Object>> entries = (List<Map<String, Object>>) project.get("source_sets");
                    if (entries == null) throw new IllegalStateException("Gradle model has no source_sets");
                    for (Map<String, Object> entry : entries) {
                        sourceSets.add(entry);
                        @SuppressWarnings("unchecked")
                        List<String> dirs = (List<String>) entry.get("output_dirs");
                        List<String> mappedOutputs = new ArrayList<>(dirs);
                        @SuppressWarnings("unchecked")
                        List<String> artifacts = (List<String>) entry.get("output_artifacts");
                        mappedOutputs.addAll(artifacts);
                        for (String dir : mappedOutputs) {
                            Object previous = outputs.putIfAbsent(dir, entry.get("id"));
                            if (previous != null && !previous.equals(entry.get("id"))) {
                                throw new IllegalStateException("Ambiguous source-set output: " + dir);
                            }
                        }
                    }
                }
                if (sourceSets.isEmpty()) {
                    throw new IllegalStateException("Gradle returned no Java source sets for " + root);
                }
                Map<String, Object> result = new LinkedHashMap<>();
                result.put("project_root", root.toString());
                result.put("gradle_version", gradleVersion);
                result.put("source_sets", sourceSets);
                result.put("output_mappings", outputs);
                result.put("mappings", mappingModels);
                return result;
            }
        } finally {
            Files.deleteIfExists(init);
            Files.deleteIfExists(modelJar);
        }
    }

    private static String groovyQuote(String value) {
        return value.replace("\\", "\\\\").replace("'", "\\'")
                .replace("\r", "\\r").replace("\n", "\\n");
    }

    public static final class ResolveAction implements BuildAction<List<String>>, Serializable {
        private static final long serialVersionUID = 1L;

        @Override public List<String> execute(BuildController controller) {
            List<String> result = new ArrayList<>();
            collect(controller, controller.getBuildModel(), new HashSet<>(), result);
            return result;
        }

        private void collect(BuildController controller, GradleBuild build,
                             Set<String> seen, List<String> result) {
            String identity = build.getBuildIdentifier().getRootDir().getAbsolutePath();
            if (!seen.add(identity)) return;
            for (BasicGradleProject project : build.getProjects()) {
                result.add(controller.getModel(project, GradleSourceModel.class).getJson());
            }
            for (GradleBuild included : build.getIncludedBuilds()) {
                collect(controller, included, seen, result);
            }
        }
    }

    private static final String INIT_SCRIPT = """
        import groovy.json.JsonOutput
        import javax.inject.Inject
        import org.gradle.api.Plugin
        import org.gradle.api.Project
        import org.gradle.api.tasks.compile.JavaCompile
        import org.gradle.tooling.provider.model.ToolingModelBuilder
        import org.gradle.tooling.provider.model.ToolingModelBuilderRegistry
        import mmm.owner.GradleSourceModel

        class MmmSourceModel implements GradleSourceModel {
            private final String json
            MmmSourceModel(String json) { this.json = json }
            String getJson() { json }
        }

        class MmmSourceModelBuilder implements ToolingModelBuilder {
            boolean canBuild(String name) { name == 'mmm.owner.GradleSourceModel' }
            Object buildAll(String name, Project project) {
                def sourceSets = project.extensions.findByName('sourceSets')
                def rows = []
                if (sourceSets != null) {
                    sourceSets.each { ss ->
                        def task = project.tasks.findByName(ss.compileJavaTaskName)
                        if (task instanceof JavaCompile) {
                            def compiler = task.javaCompiler.get()
                            def generated = task.options.generatedSourceOutputDirectory.orNull
                            def roots = ss.java.srcDirs.collect { it.canonicalPath }
                            if (generated != null) roots.add(generated.asFile.canonicalPath)
                            // JavaCompile may add files beyond SourceSet.java (plugin-generated inputs).
                            // Expose these explicitly; consumers must not silently omit them.
                            def sourceFiles = task.source.files.collect { it.canonicalPath }.sort()
                            def outputs = ss.output.classesDirs.files.collect { it.canonicalPath }
                            outputs.add(task.destinationDirectory.get().asFile.canonicalPath)
                            if (ss.output.resourcesDir != null) outputs.add(ss.output.resourcesDir.canonicalPath)
                            def archiveTask = project.tasks.findByName(ss.jarTaskName)
                            def artifacts = archiveTask instanceof org.gradle.api.tasks.bundling.Jar ?
                                [archiveTask.archiveFile.get().asFile.canonicalPath] : []
                            def buildRoot = project.rootProject.projectDir.canonicalPath
                            def id = buildRoot + ':' + project.path + ':' + ss.name
                            rows.add([
                                id: id,
                                build_root: buildRoot,
                                project_path: project.path,
                                project_dir: project.projectDir.canonicalPath,
                                name: ss.name,
                                source_roots: roots.unique().sort(),
                                source_files: sourceFiles,
                                source_includes: ss.java.includes.toList().sort(),
                                source_excludes: ss.java.excludes.toList().sort(),
                                task_source_includes: task.includes.toList().sort(),
                                task_source_excludes: task.excludes.toList().sort(),
                                classpath: task.classpath.files.collect { it.canonicalPath },
                                output_dirs: outputs.unique().sort(),
                                output_artifacts: artifacts,
                                java_home: compiler.metadata.installationPath.asFile.canonicalPath,
                                java_version: compiler.metadata.languageVersion.asInt(),
                                source_compatibility: task.sourceCompatibility,
                                target_compatibility: task.targetCompatibility,
                                release: task.options.release.orNull,
                                compiler_args: task.options.allCompilerArgs.collect { it.toString() },
                                annotation_processor_path: task.options.annotationProcessorPath == null ? [] :
                                    task.options.annotationProcessorPath.files.collect { it.canonicalPath },
                                generated_source_dirs: generated == null ? [] : [generated.asFile.canonicalPath]
                            ])
                        }
                    }
                }
                def mappingModel = null
                def loom = project.extensions.findByName('loom')
                if (loom != null) {
                    // Public Loom API supplies the already resolved mapping artifact.
                    def mappingFile = loom.getMappingsFile()
                    mappingModel = [owner: 'loom', path: null, sha256: null, namespaces: []]
                    if (mappingFile != null) {
                        def bytes = mappingFile.bytes
                        def header = new String(bytes, java.nio.charset.StandardCharsets.UTF_8).readLines().first().split('\\t')
                        if (header.length < 5 || header[0] != 'tiny' || header[1] != '2')
                            throw new IllegalStateException('Unsupported Loom mapping format')
                        mappingModel = [owner: 'loom', path: mappingFile.canonicalPath,
                            sha256: 'sha256:' + java.security.MessageDigest.getInstance('SHA-256').digest(bytes).encodeHex().toString(),
                            namespaces: header.drop(3).toList()]
                    }
                }
                new MmmSourceModel(JsonOutput.toJson([
                    gradle_version: project.gradle.gradleVersion, source_sets: rows,
                    project_dir: project.projectDir.canonicalPath, mapping_model: mappingModel]))
            }
        }

        class MmmSourceModelPlugin implements Plugin<Project> {
            private final ToolingModelBuilderRegistry registry
            @Inject MmmSourceModelPlugin(ToolingModelBuilderRegistry registry) {
                this.registry = registry
            }
            void apply(Project project) { registry.register(new MmmSourceModelBuilder()) }
        }
        gradle.beforeProject { project -> project.plugins.apply(MmmSourceModelPlugin) }
        """;
}
