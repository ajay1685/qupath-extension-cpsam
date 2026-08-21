plugins {
    // To optionally create a shadow/fat jar that bundle up any non-core dependencies
    id("com.gradleup.shadow") version "8.3.5"
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
}

// TODO: Configure your extension here (please change the defaults!)
qupathExtension {
    name = "qupath-extension-cpsam"
    group = "io.github.ajay1685"
    version = "0.2.3"
    description = "QuPath extension for Cellpose-SAM models to segment cells in 2D images via Torchscript implementation"
    automaticModule = "io.github.qupath.ext.cpsam"
}

// Dependencies
dependencies {

    // Main dependencies for most QuPath extensions
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)
    shadow(libs.qupath.fxtras)

    // DJL
    implementation(libs.deepJavaLibrary)
    implementation("io.github.qupath:qupath-extension-djl:0.4.3")

    // For testing
    testImplementation(libs.bundles.qupath)
    testImplementation(libs.junit)

}
