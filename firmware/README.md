Instructions for running the automated inference pipeline.

1. On the Windows machine, RealTimeInference.py located at C:\Users\yichao\Desktop\mintsData\raw contains the pipeline for real-time
   inference with the conformal prediction model (use RealTimeInferenceNoSurfaceReflectance.py to run the model without the surface
   reflectance bands as input features).

2. myenv and ConformalPredictionModel.pkl at the same location are the environment needed to run the script and the
   trained model respectively.

3. Make sure that all three of RealTimeInference.py, myenv and ConformalPredictionModel.pkl are in the same directory.

4. On the PowerShell, navigate to the directory containing the above.

5. Run ```myenv\Scripts\activate``` to avtivate the environment.

6. Then run the script with ```python RealTimeInference.py```.

   


   
