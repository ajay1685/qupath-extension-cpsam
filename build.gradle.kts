plugins {
    // Support writing the extension in Groovy (remove this if you don't want to)
    //groovy
    // To optionally create a shadow/fat jar that bundle up any non-core dependencies
    id("com.gradleup.shadow") version "8.3.5"
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
}

// TODO: Configure your extension here (please change the defaults!)
qupathExtension {
    name = "qupath-extension-cpsam"
    group = "io.github.ajay1685"
    version = "0.2.0"
    description = "QuPath extension for segmentating cells with CPSAM model from Cellpose-SAM"
    automaticModule = "io.github.qupath.ext.cpsam"
}

// TODO: Define your dependencies here
dependencies {

    // Main dependencies for most QuPath extensions
    shadow(libs.bundles.qupath)
    shadow(libs.bundles.logging)
    shadow(libs.qupath.fxtras)

    // If you aren't using Groovy, this can be removed
    //shadow(libs.bundles.groovy)
    implementation(libs.deepJavaLibrary)
    implementation("io.github.qupath:qupath-extension-djl:0.4.3")

    // For testing
    testImplementation(libs.bundles.qupath)
    testImplementation(libs.junit)

}
