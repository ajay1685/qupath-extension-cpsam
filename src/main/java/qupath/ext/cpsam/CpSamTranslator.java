package qupath.ext.cpsam;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * DJL Translator that converts between a single input NDArray (image tensor)
 * and the CPSAM model's multi-input format.
 *
 * Model input: (img: Tensor[BCHW], diameter, cellprob_threshold, flow_threshold, niter, batch_size)
 * Model output: Dict{"masks": Tensor[BxHxW], "flows": Tensor[Bx2xHxW], "cellprob": Tensor[BxHxW]}
 *
 * This translator packages the image NDArray with scalar parameters for the model.
 * Only the "masks" tensor is returned; flows and cellprob are discarded.
 *
 * <p>Integration notes:
 * <ol>
 *   <li>Load the model with {@code Criteria<NDArray, NDArray[]>} and
 *       {@code .optTranslator(new CpSamTranslator(...))} instead of the current
 *       raw {@code Criteria<NDList, NDList>} approach in {@code CpSam.java}.</li>
 *   <li>In {@code CpSamTileProcessor}, replace the manual NDList construction and
 *       {@code predictor.predict(NDList)} call with a typed
 *       {@code Predictor<NDArray, NDArray[]>} and pass only the image NDArray.</li>
 *   <li>Remove the manual squeeze/toType from the tile processor — this translator
 *       already normalises the mask in {@code processOutput}.</li>
 * </ol>
 */
public class CpSamTranslator implements Translator<NDArray, NDArray[]> {

    private static final Logger logger = LoggerFactory.getLogger(CpSamTranslator.class);

    private final float diameter;
    private final float cellprobThreshold;
    private final float flowThreshold;
    private final int niter;
    private final int batchSize;

    public CpSamTranslator(double diameter, float cellprobThreshold, float flowThreshold, int niter, int batchSize) {
        this.diameter = (float) diameter;
        this.cellprobThreshold = cellprobThreshold;
        this.flowThreshold = flowThreshold;
        this.niter = niter;
        this.batchSize = batchSize;
    }

    /**
     * Package the image NDArray + scalar parameters into the model's input format.
     * The model expects: (img, diameter, cellprob_threshold, flow_threshold, niter, batch_size)
     */
    @Override
    public NDList processInput(TranslatorContext ctx, NDArray input) {
        NDManager manager = ctx.getNDManager();
        boolean verboseLogging = CpSamPreferences.verboseLoggingProperty().get();

        // Ensure input is float32 for model
        NDArray img = input.toType(DataType.FLOAT32, true);

        // Create 0-d scalar tensors — TorchScript expects Python float/int scalars,
        // not 1-D arrays. manager.create(primitive) produces shape [] (scalar).
        NDArray diameterTensor = manager.create(diameter);
        NDArray cellprobTensor = manager.create(cellprobThreshold);
        NDArray flowTensor = manager.create(flowThreshold);

        // niter as int64 scalar (Groovy: manager.create(long NITER))
        NDArray niterTensor = manager.create((long) niter);
        // batch_size as int32 scalar (Groovy: manager.create(int BATCH_SIZE))
        NDArray batchSizeTensor = manager.create(batchSize);

        // Package all inputs into a single list.
        // Names must match the TorchScript model's forward() parameter names.
        NDList inputs = new NDList();
        img.setName("img");
        diameterTensor.setName("diameter");
        cellprobTensor.setName("cellprob_threshold");
        flowTensor.setName("flow_threshold");
        niterTensor.setName("niter");
        batchSizeTensor.setName("batch_size");
        inputs.add(img);
        inputs.add(diameterTensor);
        inputs.add(cellprobTensor);
        inputs.add(flowTensor);
        inputs.add(niterTensor);
        inputs.add(batchSizeTensor);

        if (verboseLogging) {
            logger.info("Translator input image: shape={}, dtype={}", img.getShape(), img.getDataType());
            logger.info("Translator scalar params: diameter={}, cellprobThreshold={}, flowThreshold={}, niter={}, batchSize={}",
                diameter, cellprobThreshold, flowThreshold, niter, batchSize);
            logger.info("Translator NDList inputs: img={}, diameterTensor={}, cellprobTensor={}, flowTensor={}, niterTensor={}, batchTensor={}",
                img.getShape(), diameterTensor.getShape(), cellprobTensor.getShape(), flowTensor.getShape(), niterTensor.getShape(), batchSizeTensor.getShape());
        }

        return inputs;
    }

    /**
     * Convert model output dict back to NDArray[] containing only the instance mask.
     * The model returns Dict{"masks": [B,H,W], "flows": [B,2,H,W], "cellprob": [B,H,W]}.
     * DJL names each NDArray in the output NDList with the dict key.
     */
    @Override
    public NDArray[] processOutput(TranslatorContext ctx, NDList output) {
        boolean verboseLogging = CpSamPreferences.verboseLoggingProperty().get();
        if (output == null || output.isEmpty()) {
            throw new IllegalStateException("CPSAM model returned no outputs");
        }

        // Extract masks by dict key; fall back to index 0 if name is not set.
        NDArray mask = output.get("masks");
        if (mask == null) mask = output.get(0);

        if (verboseLogging) {
            logger.info("Translator raw masks output: shape={}, dtype={}", mask.getShape(), mask.getDataType());
        }

        if (mask.getShape().dimension() != 3) {
            throw new IllegalStateException("CPSAM model masks output must have shape [B, H, W], got " + mask.getShape());
        }

        long batchSize = mask.getShape().get(0);
        if (batchSize != 1) {
            throw new IllegalStateException("CPSAM tile inference expects batch size 1, got " + batchSize);
        }

        // Normalize model output to a 2D mask for downstream contour extraction.
        mask = mask.squeeze(0).toType(DataType.FLOAT32, true);

        if (verboseLogging) {
            logger.info("Translator normalized mask output: shape={}, dtype={}, min={}, max={}",
                    mask.getShape(), mask.getDataType(), mask.min().getFloat(), mask.max().getFloat());
        }
        return new NDArray[]{mask};
    }
}
