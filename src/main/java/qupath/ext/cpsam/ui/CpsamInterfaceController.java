package qupath.ext.cpsam.ui;

import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.control.*;
import javafx.scene.layout.VBox;

import java.net.URL;
import java.util.ResourceBundle;

public class CpsamInterfaceController implements Initializable {

    @FXML
    private VBox root;

    @FXML
    private ComboBox<String> modelChoiceBox;

    @FXML
    private Spinner<Double> flowThresholdSpinner;

    @FXML
    private Spinner<Double> cellProbThresholdSpinner;

    @FXML
    private Spinner<Integer> niterSpinner;

    @FXML
    private Spinner<Integer> tileSizeSpinner;

    @FXML
    private Spinner<Integer> batchSizeSpinner;

    @FXML
    private Spinner<Double> normHighSpinner;

    @FXML
    private Spinner<Double> normLowSpinner;

    @FXML
    private ComboBox<String> deviceChoiceBox;

    @FXML
    private Button runButton;

    @FXML
    private Label statusLabel;

    @Override
    public void initialize(URL location, ResourceBundle resources) {
        // Initialize model choice box
        modelChoiceBox.getItems().addAll("cpsam", "todo..");
        modelChoiceBox.getSelectionModel().selectFirst();

        // Initialize device choice box
        deviceChoiceBox.getItems().addAll("CPU", "GPU");
        deviceChoiceBox.getSelectionModel().selectFirst();

        // Set up spinners with default values
        flowThresholdSpinner.getValueFactory().setValue(1.0);
        cellProbThresholdSpinner.getValueFactory().setValue(0.5);
        niterSpinner.getValueFactory().setValue(10);
        tileSizeSpinner.getValueFactory().setValue(256);
        batchSizeSpinner.getValueFactory().setValue(8);


        // Set up run button action
        runButton.setOnAction(event -> runSegmentation());
    }

    private void runSegmentation() {
        // Get values from controls
        String model = modelChoiceBox.getValue();
        Double flowThreshold = flowThresholdSpinner.getValueFactory().getValue();
        Double cellProbThreshold = cellProbThresholdSpinner.getValueFactory().getValue();
        Integer niter = niterSpinner.getValueFactory().getValue();
        Integer tileSize = tileSizeSpinner.getValueFactory().getValue();
        Integer batchSize = batchSizeSpinner.getValueFactory().getValue();

        String device = deviceChoiceBox.getValue();

        // Update status label
        statusLabel.setText("Running segmentation with: " + model + ", " + device);

        // In a real implementation, you would pass these parameters to your segmentation algorithm
        System.out.println("Parameters:");
        System.out.println("Model: " + model);
        System.out.println("Flow Threshold: " + flowThreshold);
        System.out.println("Cell Probability Threshold: " + cellProbThreshold);
        System.out.println("Niter: " + niter);
        System.out.println("Tile Size: " + tileSize);
        System.out.println("Batch Size: " + batchSize);

        System.out.println("Device: " + device);
    }
}