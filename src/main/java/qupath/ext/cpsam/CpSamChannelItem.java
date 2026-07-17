package qupath.ext.cpsam;

import qupath.lib.color.ColorDeconvolutionStains;
import qupath.lib.images.servers.ColorTransforms;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * Represents a selectable channel for CPSAM preprocessing.
 * <p>
 * Supports both raw image channels and color deconvolution channels.
 * Used to populate the three channel ComboBoxes in the UI and to
 * produce {@link ColorTransforms.ColorTransform} objects for the
 * preprocessing pipeline.
 */
public class CpSamChannelItem {

    private static final String CHANNEL_NONE = "(None)";

    /** Display name shown in the ComboBox dropdown. */
    private final String displayName;
    /** Underlying channel name or index used to create the ColorTransform. */
    private final String identifier;
    /** Channel index (0-based) for raw channels, -1 for deconvolution or None. */
    private final int index;
    /** Deconvolution stain number (1-based), 0 if not a deconvolution channel. */
    private final int stainNumber;
    /** Deconvolution stains object, null if not applicable. */
    private final ColorDeconvolutionStains stains;
    /** Whether this represents a "(None)" zero-pad option. */
    private final boolean isNone;

    CpSamChannelItem(String displayName, String identifier, int index, int stainNumber, ColorDeconvolutionStains stains, boolean isNone) {
        this.displayName = displayName;
        this.identifier = identifier;
        this.index = index;
        this.stainNumber = stainNumber;
        this.stains = stains;
        this.isNone = isNone;
    }

    @Override
    public String toString() {
        return displayName;
    }

    /**
     * Create a {@link ColorTransforms.ColorTransform} for this channel item.
     * @return the ColorTransform, or null if this is a "(None)" item
     */
    ColorTransforms.ColorTransform getTransform() {
        if (isNone) {
            return null;
        }
        if (stainNumber > 0) {
            // Color deconvolution channel
            return ColorTransforms.createColorDeconvolvedChannel(stains, stainNumber);
        }
        // Raw channel by index
        return ColorTransforms.createChannelExtractor(index);
    }

    /**
     * Check if this item represents a "(None)" zero-pad option.
     */
    public boolean isNone() {
        return isNone;
    }

    /**
     * Get the channel index (0-based) for raw channels.
     */
    public int getIndex() {
        return index;
    }

    /**
     * Get the stain number (1-based) for deconvolution channels.
     */
    public int getStainNumber() {
        return stainNumber;
    }

    /**
     * Get the display name.
     */
    public String getDisplayName() {
        return displayName;
    }

    /**
     * Get the identifier used to create the ColorTransform.
     */
    public String getIdentifier() {
        return identifier;
    }

    /**
     * Get available channel items for the given image data.
     * <p>
     * Includes raw channels, "(None)" options for optional slots,
     * and color deconvolution channels if available.
     */
    public static List<CpSamChannelItem> getAvailableChannels(qupath.lib.images.ImageData<?> imageData) {
        var stains = imageData.getColorDeconvolutionStains();
        List<CpSamChannelItem> list = new ArrayList<>();
        var server = imageData.getServer();
        var metadata = server.getMetadata();
        int nChannels = metadata.getSizeC();

        // Add raw channels
        for (int i = 0; i < nChannels; i++) {
            var channel = metadata.getChannels().get(i);
            String name = channel.getName();
            list.add(new CpSamChannelItem(name, name, i, 0, stains, false));
        }

        // Add color deconvolution channels if stains are available
        if (stains != null) {
            for (int s = 1; s <= 3; s++) {
                if (!stains.getStain(s).isResidual()) {
                    String stainName = stains.getStain(s).getName();
                    String displayName = stainName;
                    list.add(new CpSamChannelItem(displayName, stainName, -1, s, stains, false));
                }
            }
        }

        return list;
    }

    /**
     * Get "(None)" channel items for optional channel slots.
     */
    public static List<CpSamChannelItem> getNoneItems() {
        return List.of(new CpSamChannelItem(CHANNEL_NONE, CHANNEL_NONE, -1, 0, null, true));
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof CpSamChannelItem that)) return false;
        return index == that.index && stainNumber == that.stainNumber && isNone == that.isNone;
    }

    @Override
    public int hashCode() {
        return Objects.hash(index, stainNumber, isNone);
    }
}
