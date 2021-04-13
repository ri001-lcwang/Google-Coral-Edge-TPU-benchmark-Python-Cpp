## Install software package
1. Install edge-tpu-runtime
Follow up https://coral.ai/docs/accelerator/get-started/#1-install-the-edge-tpu-runtime

2. Install tensorflow 2.1.0
cd ~
wget https://github.com/PINTO0309/Tensorflow-bin/blob/master/tensorflow-2.1.0-cp37-cp37m-linux_armv7l_download.sh
tensorflow-2.1.0-cp37-cp37m-linux_armv7l_download.sh
pip3 install tensorflow-2.1.0-cp37-cp37m-linux_armv7l.whl

3. Install tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl
cd ~
pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-linux_armv7l.whl

#Note: To guarantee running without 
#"RuntimeError: Internal: Unsupported data type in custom op handler: 24348008Node number 2 (EdgeTpuDelegateForCustomOp) failed to prepare.",
#package version is as the following.
pi@raspberrypi:~ $ pip3 list
Package                Version
---------------------- ------------
edgetpu                2.14.1
tensorflow             2.1.0
tensorflow-estimator   2.1.0
termcolor              1.1.0
tflite-runtime         2.1.0.post1



## Run
python3 Detection_tpu_benchmark.py


## Note: It is implemented with Auto detect Edgetpu device.
# Result while TPU is connected.
pi@raspberrypi:~/Desktop/hawkeye/stanley_test_files $ python3 Detection_tpu_benchmark.py
/usr/local/lib/python3.7/dist-packages/numba/errors.py:137: UserWarning: Insufficiently recent colorama version found. Numba requires colorama >= 0.3.9
  warnings.warn(msg)
Note: Detect edgetpu devices is usb. Run TPU & CPU.

  Benchmark result:
     Device    | Opt model | Data type | Pred_threads | Accuracy(%) | Load model(s) | 1st pred(s)  | Avg 17 pred(s) | Avg 17 set_io + pred(s)
  TPU, runtime |     N     |    int8   |      1       |    70.59    |    2.91156    |    0.04517   |     0.02495    |       0.02571
  TPU, runtime |     Y     |    int8   |      1       |    76.47    |    0.09077    |    0.04526   |     0.02488    |       0.02564
  TPU, runtime |     N     |   uint8   |      1       |    64.71    |    0.09036    |    0.04534   |     0.02485    |       0.02558
  TPU, runtime |     Y     |   uint8   |      1       |    82.35    |    0.09051    |    0.04543   |     0.02471    |       0.02543
  CPU, tflite  |     N     |    int8   |      1       |    76.47    |     0.0073    |    0.31405   |     0.30692    |       0.30772
  CPU, tflite  |     Y     |    int8   |      1       |    76.47    |    0.00233    |    0.31203   |     0.30146    |       0.30228
  CPU, tflite  |     N     |   uint8   |      1       |    82.35    |    0.00236    |    0.31162   |     0.30127    |       0.30205
  CPU, tflite  |     Y     |   uint8   |      1       |    76.47    |    0.00239    |    0.31088   |     0.30456    |       0.30534
  CPU, tflite  |     N     |     fp16  |      1       |    82.35    |    0.00157    |    0.89004   |     0.31953    |       0.32033
  CPU, tflite  |     Y     |     fp16  |      1       |    82.35    |    0.00175    |    1.0413    |     0.32855    |       0.32933
  CPU, tflite  |     N     |  fp32_fb  |      1       |    82.35    |    0.00157    |    0.88657   |     0.31897    |       0.31977
  CPU, tflite  |     Y     |  fp32_fb  |      1       |    76.47    |    0.00235    |    0.32367   |     0.31355    |       0.3144
  CPU, tflite  |     N     |     fp32  |      1       |    82.35    |    0.00165    |    0.90125   |     0.32086    |       0.32164
  CPU, tflite  |     Y     |     fp32  |      1       |    82.35    |    0.00163    |    0.49759   |     0.49229    |       0.49304
# => Analysis.
# => Best result.
# TPU, runtime |     Y     |   uint8   |      1       |    82.35    |    0.09051    |    0.04543   |     0.02471    |       0.02543
# CPU, tflite  |     N     |   uint8   |      1       |    82.35    |    0.00236    |    0.31162   |     0.30127    |       0.30205
# => Speedup for Avg 17 set_io + pred = 11.9
# CPU, tflite  |     N     |     fp32  |      1       |    82.35    |    0.00165    |    0.90125   |     0.32086    |       0.32164 
# => Speedup for Avg 17 set_io + pred = 12.6



# Result while TPU is not connected.
pi@raspberrypi:~/Desktop/hawkeye/stanley_test_files $ python3 Detection_tpu_benchmark.py
/usr/local/lib/python3.7/dist-packages/numba/errors.py:137: UserWarning: Insufficiently recent colorama version found. Numba requires colorama >= 0.3.9
  warnings.warn(msg)
Warning: Detect edgetpu devices is none. Run without TPU.

  Benchmark result:
     Device    | Opt model | Data type | Pred_threads | Accuracy(%) | Load model(s) | 1st pred(s)  | Avg 17 pred(s) | Avg 17 set_io + pred(s)
  CPU, tflite  |     N     |    int8   |      1       |    76.47    |    0.00991    |    0.30811   |     0.29742    |       0.29825
  CPU, tflite  |     Y     |    int8   |      1       |    76.47    |    0.00217    |    0.31583   |     0.30457    |       0.30538
  CPU, tflite  |     N     |   uint8   |      1       |    82.35    |    0.00217    |    0.31503   |     0.31774    |       0.31859
  CPU, tflite  |     Y     |   uint8   |      1       |    76.47    |    0.00215    |    0.31536   |     0.31128    |       0.31216
  CPU, tflite  |     N     |     fp16  |      1       |    82.35    |    0.00143    |    0.8711    |     0.3176     |       0.31836
  CPU, tflite  |     Y     |     fp16  |      1       |    82.35    |    0.00163    |    1.05617   |     0.32797    |       0.32875
  CPU, tflite  |     N     |  fp32_fb  |      1       |    82.35    |    0.00137    |    0.88168   |     0.35624    |       0.35711
  CPU, tflite  |     Y     |  fp32_fb  |      1       |    76.47    |    0.00223    |    0.32255   |     0.30734    |       0.30819
  CPU, tflite  |     N     |     fp32  |      1       |    82.35    |    0.00141    |    0.86534   |     0.31414    |       0.3149
  CPU, tflite  |     Y     |     fp32  |      1       |    82.35    |    0.00145    |    0.51334   |     0.4841     |       0.48487

