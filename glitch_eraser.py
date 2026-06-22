import numpy as np
import cv2
import scipy.ndimage as ndimage

class SelfHealingGlitchEraser:
    def __init__(self, cpr_max=3.0, dop_max=1.0):
        self.cpr_max = cpr_max
        self.dop_max = dop_max

    def detect_glitches(self, matrix, layer_type="cpr"):
        """
        Detects anomalies and generates a binary mask where:
        1 = Glitched/Corrupted pixel (needs healing)
        0 = Clean pixel
        """
        # Create initial mask for NaNs and infinities
        mask = np.isnan(matrix) | np.isinf(matrix)
        
        # Check zeros (often indicate raw telemetry drops/no-signal lines)
        mask = mask | (matrix == 0.0)
        
        # Out of bounds checks
        if layer_type == "cpr":
            mask = mask | (matrix > self.cpr_max) | (matrix < 0.01)
        elif layer_type == "dop":
            mask = mask | (matrix > self.dop_max) | (matrix < 0.01)
            
        return mask.astype(np.uint8)

    def heal_layer(self, matrix, layer_type="cpr", inpaint_radius=3):
        """
        Self-heals the matrix on the fly using a combined pipeline of:
        1. Anomaly detection (masking)
        2. OpenCV Inpainting for structural/line drops
        3. Median filter smoothing for residual salt-and-pepper noise
        """
        # 1. Detect glitches
        glitch_mask = self.detect_glitches(matrix, layer_type=layer_type)
        
        # If no glitches, return original
        if np.sum(glitch_mask) == 0:
            return matrix.copy(), glitch_mask
            
        # Create a working copy and clean NaNs to allow CV2 processing
        clean_working = matrix.copy()
        clean_working[np.isnan(clean_working) | np.isinf(clean_working)] = 0.0
        
        # We need to normalize or scale for CV2 inpainting (OpenCV inpaint operates on 8-bit or 32-bit float images)
        # Since our matrices are float32, we can pass them directly to CV2 inpaint
        clean_working = clean_working.astype(np.float32)
        
        # 2. OpenCV Inpainting (using Fast Marching Method - Telea)
        # cv2.inpaint requires a 1-channel 8-bit mask where pixels to be inpainted are > 0
        healed_cv2 = cv2.inpaint(clean_working, glitch_mask, inpaint_radius, cv2.INPAINT_TELEA)
        
        # 3. Post-process remaining local spikes using an adaptive median filter (only on repaired locations)
        # This keeps the original sharp details of the terrain intact while ensuring smooth transitions for repaired pixels
        median_filtered = ndimage.median_filter(healed_cv2, size=3)
        
        # Merge: keep original pixels where they were healthy, use healed values where they were glitched
        healed_final = np.where(glitch_mask == 1, median_filtered, healed_cv2)
        
        # Final physical bounds clip
        if layer_type == "cpr":
            healed_final = np.clip(healed_final, 0.01, self.cpr_max)
        elif layer_type == "dop":
            healed_final = np.clip(healed_final, 0.01, self.dop_max)
            
        return healed_final, glitch_mask

    def heal_telemetry_packet(self, dataset):
        """
        Heals an entire dictionary of simulated lunar data layers.
        """
        healed_dataset = dataset.copy()
        
        # Heal CPR
        cpr_healed, cpr_mask = self.heal_layer(dataset["cpr"], layer_type="cpr")
        healed_dataset["cpr"] = cpr_healed
        healed_dataset["cpr_mask"] = cpr_mask
        
        # Heal DOP
        dop_healed, dop_mask = self.heal_layer(dataset["dop"], layer_type="dop")
        healed_dataset["dop"] = dop_healed
        healed_dataset["dop_mask"] = dop_mask
        
        # Re-evaluate ice presence on cleaned data
        # (Ice is detected where CPR > 1.0 and DOP < 0.13)
        # Note: real lunar regolith has CPR < 0.4 and DOP > 0.6
        return healed_dataset
