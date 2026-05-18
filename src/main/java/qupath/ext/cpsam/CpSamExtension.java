package qupath.ext.cpsam;

import javafx.beans.property.BooleanProperty;
import javafx.beans.property.Property;
import javafx.beans.property.StringProperty;
import javafx.scene.Scene;
import javafx.scene.control.MenuItem;
import javafx.scene.layout.BorderPane;
import javafx.stage.Stage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import qupath.ext.cpsam.ui.CpSamInterfaceController;
import qupath.fx.prefs.controlsfx.PropertyItemBuilder;
import qupath.fx.utils.FXUtils;
import qupath.lib.common.Version;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.extensions.GitHubProject;
import qupath.lib.gui.extensions.QuPathExtension;
import qupath.lib.gui.prefs.PathPrefs;

import java.io.IOException;
import java.util.ResourceBundle;


public class CpSamExtension implements QuPathExtension, GitHubProject {
	private static final Logger logger = LoggerFactory.getLogger(CpSamExtension.class);
	private static final ResourceBundle resources = ResourceBundle.getBundle("qupath.ext.cpsam.ui.strings");

	private static final String EXTENSION_NAME = resources.getString("name");
	private static final String EXTENSION_DESCRIPTION = resources.getString("description");

	private static final Version EXTENSION_QUPATH_VERSION = Version.parse("v0.7.0");

	private static final GitHubRepo EXTENSION_REPOSITORY = GitHubRepo.create(
			"CPSAM", "ajay1685", "qupath-extension-cpsam");

	private boolean isInstalled = false;

	private final BooleanProperty enableExtensionProperty = PathPrefs.createPersistentPreference(
			"cpsam.enableExtension", true);

	private Stage stage;

	@Override
	public void installExtension(QuPathGUI qupath) {
		if (isInstalled) {
			logger.debug("{} is already installed", getName());
			return;
		}
		isInstalled = true;
		addMenuItem(qupath);
		addPreferenceToPane(qupath);
	}

	private void addPreferenceToPane(QuPathGUI qupath) {
		var propertyItem = new PropertyItemBuilder<>(enableExtensionProperty, Boolean.class)
				.name(resources.getString("menu.enable"))
				.category("CPSAM extension")
				.description("Enable CPSAM extension")
				.build();
		var verboseLoggingItem = new PropertyItemBuilder<>(CpSamPreferences.verboseLoggingProperty(), Boolean.class)
				.name("Verbose logging")
				.category("CPSAM extension")
				.description("Log detailed CPSAM pipeline diagnostics for debugging")
				.build();
		qupath.getPreferencePane()
				.getPropertySheet()
				.getItems()
				.add(propertyItem);
		qupath.getPreferencePane()
				.getPropertySheet()
				.getItems()
				.add(verboseLoggingItem);
	}


	private void addMenuItem(QuPathGUI qupath) {
		var menu = qupath.getMenu("Extensions>" + EXTENSION_NAME, true);
		MenuItem menuItem = new MenuItem("Run CPSAM Segmentation");
		menuItem.setOnAction(e -> createStage(qupath));
		menuItem.disableProperty().bind(enableExtensionProperty.not());
		menu.getItems().add(menuItem);
	}

	private void createStage(QuPathGUI qupath) {
		if (stage == null) {
			try {
				stage = new Stage();
				var pane = CpSamInterfaceController.createInstance(qupath);
				Scene scene = new Scene(new BorderPane(pane));
				stage.setScene(scene);
				stage.initOwner(QuPathGUI.getInstance().getStage());
				stage.setTitle(resources.getString("title"));
				stage.setResizable(true);
				// Free cached GPU model when the panel is closed
				stage.setOnHidden(e -> CpSam.clearModelCache());
			} catch (IOException e) {
				logger.error("Failed to load CPSAM UI", e);
				return;
			}
		}
		stage.show();
		stage.sizeToScene();
		if (stage.isShowing() && Double.isFinite(stage.getX()) && Double.isFinite(stage.getY()))
			FXUtils.retainWindowPosition(stage);
	}


	@Override
	public String getName() {
		return EXTENSION_NAME;
	}

	@Override
	public String getDescription() {
		return EXTENSION_DESCRIPTION;
	}

	@Override
	public Version getQuPathVersion() {
		return EXTENSION_QUPATH_VERSION;
	}

	@Override
	public GitHubRepo getRepository() {
		return EXTENSION_REPOSITORY;
	}
}
