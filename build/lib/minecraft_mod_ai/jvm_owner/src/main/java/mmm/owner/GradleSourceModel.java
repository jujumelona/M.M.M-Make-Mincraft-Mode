package mmm.owner;

import java.io.Serializable;

/** Small shared interface loaded by both the Tooling API client and Gradle daemon. */
public interface GradleSourceModel extends Serializable {
    String getJson();
}
