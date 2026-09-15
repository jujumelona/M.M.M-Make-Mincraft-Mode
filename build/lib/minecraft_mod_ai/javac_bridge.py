"""JDK Compiler Tree API bridge. Annotation processing is always disabled."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


JAVA_SOURCE = r'''
import java.net.URI;
import java.nio.file.*;
import java.util.*;
import javax.tools.*;
import javax.lang.model.element.*;
import javax.lang.model.type.*;
import javax.lang.model.util.*;
import com.sun.source.tree.*;
import com.sun.source.util.*;

class IntegritySymbols {
  static Types types; static Elements elements;
  static String q(String s) {
    StringBuilder b = new StringBuilder("\"");
    for (char c : s.toCharArray()) {
      if (c == '\\' || c == '"') b.append('\\').append(c);
      else if (c < 32) b.append(String.format("\\u%04x", (int)c));
      else b.append(c);
    }
    return b.append('"').toString();
  }
  static String desc(TypeMirror raw) {
    TypeMirror t = types.erasure(raw);
    switch(t.getKind()) {
      case BOOLEAN: return "Z"; case BYTE: return "B"; case SHORT: return "S";
      case INT: return "I"; case LONG: return "J"; case CHAR: return "C";
      case FLOAT: return "F"; case DOUBLE: return "D"; case VOID: return "V";
      case ARRAY: return "[" + desc(((ArrayType)t).getComponentType());
      case DECLARED: return "L" + elements.getBinaryName((TypeElement)((DeclaredType)t).asElement()).toString().replace('.', '/') + ";";
      default: throw new IllegalArgumentException("Unresolved type: " + t);
    }
  }
  static String descriptor(Element e) {
    if (e instanceof ExecutableElement) {
      ExecutableElement m = (ExecutableElement)e;
      StringBuilder b = new StringBuilder("(");
      for (VariableElement p : m.getParameters()) b.append(desc(p.asType()));
      return b.append(')').append(e.getKind()==ElementKind.CONSTRUCTOR ? "V" : desc(m.getReturnType())).toString();
    }
    return desc(e.asType());
  }
  static String side(Element e) {
    for (Element current=e; current!=null; current=current.getEnclosingElement()) {
      for (AnnotationMirror a:current.getAnnotationMirrors()) {
        String name=a.getAnnotationType().toString();
        if(name.equals("net.fabricmc.api.Environment") || name.endsWith(".distmarker.OnlyIn")) {
          for (AnnotationValue value: a.getElementValues().values()) {
            String v=value.getValue().toString();
            if(v.equals("CLIENT")) return "CLIENT";
            if(v.equals("SERVER") || v.equals("DEDICATED_SERVER")) return "SERVER";
          }
          throw new IllegalArgumentException("Unresolved side annotation: " + a);
        }
      }
    }
    return "COMMON";
  }
  public static void main(String[] args) throws Exception {
    String mode=args[0], release=args[1], cp=args[2], filename=args[3];
    String code = Files.readString(Path.of(args[4]));
    JavaCompiler compiler = ToolProvider.getSystemJavaCompiler();
    if (compiler == null) throw new IllegalStateException("JDK compiler required");
    DiagnosticCollector<JavaFileObject> diagnostics = new DiagnosticCollector<>();
    JavaFileObject input = new SimpleJavaFileObject(URI.create("string:///" + filename), JavaFileObject.Kind.SOURCE) {
      public CharSequence getCharContent(boolean ignore) { return code; }
    };
    try (StandardJavaFileManager fm=compiler.getStandardFileManager(diagnostics, null, null)) {
      JavacTask task=(JavacTask)compiler.getTask(null,fm,diagnostics,
          List.of("-proc:none","--release",release,"-classpath",cp),null,List.of(input));
      List<CompilationUnitTree> units=new ArrayList<>();
      task.parse().forEach(units::add);
      if (!mode.equals("parse")) task.analyze();
      boolean failed=false;
      for (Diagnostic<?> d: diagnostics.getDiagnostics()) if(d.getKind()==Diagnostic.Kind.ERROR) {
        System.err.println(d.toString()); failed=true;
      }
      if(failed) System.exit(2);
      if(mode.equals("parse")) { System.out.println("[]"); return; }
      Trees trees=Trees.instance(task); types=task.getTypes(); elements=task.getElements();
      List<String> rows=new ArrayList<>();
      for(CompilationUnitTree unit: units) new TreePathScanner<Void,Void>() {
        void emit(TreePath path, Tree tree) {
          Element e=trees.getElement(path);
          if(e==null) throw new IllegalArgumentException("Unresolved reference: " + tree);
          ElementKind k=e.getKind();
          if(!(k==ElementKind.METHOD || k==ElementKind.CONSTRUCTOR || k==ElementKind.FIELD || k==ElementKind.ENUM_CONSTANT || e instanceof TypeElement)) return;
          TypeElement owner=e instanceof TypeElement ? (TypeElement)e : (TypeElement)e.getEnclosingElement();
          String kind=e instanceof TypeElement ? "CLASS" : (k==ElementKind.CONSTRUCTOR?"CONSTRUCTOR":(e instanceof ExecutableElement?"METHOD":"FIELD"));
          String name=k==ElementKind.CONSTRUCTOR?"<init>":e.getSimpleName().toString();
          long pos=trees.getSourcePositions().getStartPosition(unit,tree);
          rows.add("{\"owner\":"+q(elements.getBinaryName(owner).toString().replace('.','/'))+
            ",\"name\":"+q(name)+",\"descriptor\":"+q(descriptor(e))+",\"kind\":"+q(kind)+",\"side\":"+q(side(e))+
            ",\"is_static\":"+e.getModifiers().contains(Modifier.STATIC)+",\"line\":"+unit.getLineMap().getLineNumber(pos)+"}");
        }
        public Void visitMethodInvocation(MethodInvocationTree t,Void v) {
          emit(new TreePath(getCurrentPath(),t.getMethodSelect()),t); return super.visitMethodInvocation(t,v);
        }
        public Void visitNewClass(NewClassTree t,Void v) { emit(getCurrentPath(),t); return super.visitNewClass(t,v); }
        public Void visitMemberReference(MemberReferenceTree t,Void v) { emit(getCurrentPath(),t); return super.visitMemberReference(t,v); }
        public Void visitMemberSelect(MemberSelectTree t,Void v) {
          Element e=trees.getElement(getCurrentPath());
          if(e!=null && (e instanceof TypeElement || e.getKind()==ElementKind.FIELD || e.getKind()==ElementKind.ENUM_CONSTANT)) emit(getCurrentPath(),t);
          return super.visitMemberSelect(t,v);
        }
        public Void visitIdentifier(IdentifierTree t,Void v) {
          Element e=trees.getElement(getCurrentPath());
          if(e!=null && (e instanceof TypeElement || e.getKind()==ElementKind.FIELD || e.getKind()==ElementKind.ENUM_CONSTANT)) emit(getCurrentPath(),t);
          return super.visitIdentifier(t,v);
        }
      }.scan(unit,null);
      System.out.println("["+String.join(",",rows)+"]");
    }
  }
}
'''


def analyze_java(source: str, *, classpath=(), java_version="17", filename="IntegrityInput.java",
                 parse_only=False, timeout=60) -> list[dict]:
    java = shutil.which("java")
    if not java:
        raise RuntimeError("JAVA_TOOLCHAIN_UNAVAILABLE")
    paths = [Path(p).resolve() for p in classpath]
    if any(not p.exists() for p in paths):
        raise ValueError("JAVA_CLASSPATH_MISSING")
    if Path(filename).name != filename or not filename.endswith(".java"):
        raise ValueError("JAVA_FILENAME_INVALID")
    with tempfile.TemporaryDirectory(prefix="mmm-javac-") as temp:
        root = Path(temp)
        helper = root / "IntegritySymbols.java"
        helper.write_text(JAVA_SOURCE, encoding="utf-8")
        input_path = root / "source.txt"
        input_path.write_text(source, encoding="utf-8")
        # Empty classpath must not implicitly resolve classes from the current directory.
        cp = os.pathsep.join(map(str, paths)) or str(root / "empty")
        result = subprocess.run([java, str(helper), "parse" if parse_only else "resolve",
                                 str(java_version), cp, filename, str(input_path)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace",
                                timeout=timeout, cwd=root)
        if result.returncode:
            raise ValueError("JAVA_ANALYSIS_FAILED: " + result.stderr[-12000:])
        return json.loads(result.stdout)
