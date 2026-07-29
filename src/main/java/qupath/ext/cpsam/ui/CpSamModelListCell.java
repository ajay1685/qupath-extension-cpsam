package qupath.ext.cpsam.ui;

import javafx.scene.control.ListCell;
import javafx.scene.control.Tooltip;
import qupath.ext.cpsam.CpSamModel;


/**
 * Custom list cell for displaying model variants in a ComboBox.
 * Shows the model display name with status icon, device tag, and version.
 * Undownloaded models are rendered in muted color.
 * Shows "update available" in tooltip when catalog version > cached downloaded version.
 */
public class CpSamModelListCell extends ListCell<CpSamModel> {

    private final Tooltip tooltip;

    public CpSamModelListCell() {
        super();
        tooltip = new Tooltip();
    }

    @Override
    public void updateItem(CpSamModel model, boolean empty) {
        super.updateItem(model, empty);
        if (empty || model == null) {
            setText(null);
            setStyle("");
            setTooltip(null);
            return;
        }

        // Build label: [status] displayName [device] (version)
        String statusIcon = model.isValid() ? "✓ " : "⛔ ";  // checkmark or tombstone
        String name = model.getDisplayName();
        String deviceTag = getDeviceTag(model.getDeviceConstraint());
        String version = model.getVersion() != null && !model.getVersion().isEmpty() ? " (" + model.getVersion() + ")" : "";
        String label = statusIcon + name + deviceTag + version;

        setText(label);

        // Color: muted for not-downloaded
        if (!model.isValid()) {
            setStyle("-fx-text-fill: #999999;");
        } else {
            setStyle("");
        }

        // Tooltip: description + path + device constraint + update check
        StringBuilder tip = new StringBuilder();
        String desc = model.getDescription();
        if (!desc.isEmpty()) {
            tip.append(desc);
        }
        if (model.isValid() && model.getPath() != null) {
            if (tip.length() > 0) tip.append("\n");
            tip.append("Path: ").append(model.getPath().toAbsolutePath());

            // Check if catalog version > cached downloaded version
            String catalogVersion = model.getVersion();
            if (catalogVersion != null && !catalogVersion.isEmpty()) {
                String cachedVersion = CpSamPreferences.downloadedVersionProperty(
                        model.getName().orElse("")).get();
                if (cachedVersion != null && !cachedVersion.equals(catalogVersion)) {
                    tip.append("\nUpdate available: ").append(cachedVersion).append(" → ").append(catalogVersion);
                }
            }
        } else {
            if (tip.length() > 0) tip.append("\n");
            tip.append("Not downloaded — click Download to fetch");
        }
        tooltip.setText(tip.toString());
        setTooltip(tooltip);
    }

    private String getDeviceTag(CpSamRemoteModel.DeviceConstraint constraint) {
        return switch (constraint) {
            case ANY -> "";
            case CPU -> " [CPU]";
            case CUDA -> " [CUDA]";
        };
    }
}
