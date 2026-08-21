pluginManagement {
    repositories {
        gradlePluginPortal()
        maven {
            url = uri("https://maven.scijava.org/content/repositories/releases")
        }
    }
}

// Targets QuPath 0.7.0 but uses Measurement API from 0.8.0 if available.
qupath {
    version = "0.7.0"
}

// Apply QuPath Gradle settings plugin to handle configuration
plugins {
    id("io.github.qupath.qupath-extension-settings") version "0.2.1"
}
