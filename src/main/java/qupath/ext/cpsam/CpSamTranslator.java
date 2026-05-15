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
 * Model output: [mask: Tensor[1xHxW]]
 *
 * This translator packages the image NDArray with scalar parameters for the model.
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

        // Package all inputs into a single list
        NDList inputs = new NDList();
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
     * Convert model output (mask tensor) back to NDArray[].
     */
    @Override
        public NDArray[] processOutput(TranslatorContext ctx, NDList output) {
            boolean verboseLogging = CpSamPreferences.verboseLoggingProperty().get();
            if (output == null || output.isEmpty()) {
                throw new IllegalStateException("CPSAM model returned no outputs");
            }

            NDArray mask = output.get(0);

            if (verboseLogging) {
                logger.info("Translator raw output[0]: shape={}, dtype={}", mask.getShape(), mask.getDataType());
            }

            if (mask.getShape().dimension() != 3) {
                throw new IllegalStateException("CPSAM model output must have shape [B, H, W], got " + mask.getShape());
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
