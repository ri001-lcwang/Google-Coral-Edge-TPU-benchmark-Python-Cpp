#https://github.com/mattroos/EdgeTpuTesting/blob/master/test_edgetpu.ipynb
#https://www.tensorflow.org/lite/performance/post_training_quantization#integer_only
#https://zhuanlan.zhihu.com/p/79744430

import tensorflow as tf
filename = 'freesound_umedia_20_classes_skip_attention_model_0420.h5'
model = tf.keras.models.load_model(filename)

for j in range(0, 10):
#for j in range(6, 10):
    if j==0:
        print('')
        print('convert count= ', j+1)
        print('convert tflite model')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        '''
        from tensorflow.lite.python.lite import TFLiteConverter, TFLiteConverterV2
        converter = TFLiteConverterV2.from_keras_model(model)
        '''
        tflite_model = converter.convert()
        open("freesound_umedia_20_classes_skip_attention_model_0420.tflite", "wb").write(tflite_model)

    elif j==1:
        #add opt
        print('')
        print('convert count= ', j+1)
        print('convert tflite opt model')
        converter_1 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_1.optimizations = [tf.lite.Optimize.DEFAULT]
        '''
        from tensorflow.lite.python.lite import TFLiteConverter, TFLiteConverterV2
        converter_1 = TFLiteConverterV2.from_keras_model(model)
        '''
        tflite_model = converter_1.convert()
        open("freesound_umedia_20_classes_skip_attention_model_0420_opt.tflite", "wb").write(tflite_model)
        
    elif j==2:
        print('')
        print('convert count= ', j+1)
        print('convert tflite model for coral tpu (int8)')
        converter_2 = tf.lite.TFLiteConverter.from_keras_model(model)
        image_shape = (126, 64, 1)
        '''
        def representative_dataset_gen():
            num_calibration_steps = 1
            for _ in range(num_calibration_steps):
                image = tf.random.normal([1] + list(image_shape))
                yield [input]
        '''
        def representative_dataset_gen():
            num_calibration_images = 100
            for i in range(num_calibration_images):
        #         image = tf.random.normal([1] + list(image_shape))
                image = tf.random.uniform([1] + list(image_shape))
                yield [image]

        converter_2.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
        converter_2.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_2.inference_input_type = tf.int8  # or tf.uint8
        converter_2.inference_output_type = tf.int8  # or tf.uint8
        tflite_quant_model = converter_2.convert()
        open("freesound_umedia_20_classes_skip_attention_model_0420_int8.tflite", "wb").write(tflite_quant_model)

    elif j==3:
        #add opt
        print('')
        print('convert count= ', j+1)
        print('convert tflite opt model for coral tpu (int8)')
        converter_3 = tf.lite.TFLiteConverter.from_keras_model(model)
        image_shape = (126, 64, 1)
        '''
        def representative_dataset_gen():
            num_calibration_steps = 1
            for _ in range(num_calibration_steps):
                image = tf.random.normal([1] + list(image_shape))
                yield [input]
        '''
        def representative_dataset_gen():
            num_calibration_images = 100
            for i in range(num_calibration_images):
        #         image = tf.random.normal([1] + list(image_shape))
                image = tf.random.uniform([1] + list(image_shape))
                yield [image]

        converter_3.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_3.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
        converter_3.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_3.inference_input_type = tf.int8  # or tf.uint8
        converter_3.inference_output_type = tf.int8  # or tf.uint8
        tflite_quant_model = converter_3.convert()
        open("freesound_umedia_20_classes_skip_attention_model_0420_opt_int8.tflite", "wb").write(tflite_quant_model)
        
    elif j==4:
        print('')
        print('convert count= ', j+1)
        print('convert tflite model for coral tpu (uint8)')
        converter_4 = tf.lite.TFLiteConverter.from_keras_model(model)
        image_shape = (126, 64, 1)
        '''
        def representative_dataset_gen():
            num_calibration_steps = 1
            for _ in range(num_calibration_steps):
                image = tf.random.normal([1] + list(image_shape))
                yield [input]
        '''
        def representative_dataset_gen():
            num_calibration_images = 100
            for i in range(num_calibration_images):
        #         image = tf.random.normal([1] + list(image_shape))
                image = tf.random.uniform([1] + list(image_shape))
                yield [image]

        converter_4.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
        converter_4.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_4.inference_input_type = tf.uint8
        converter_4.inference_output_type = tf.uint8
        tflite_quant_model = converter_4.convert()
        open("freesound_umedia_20_classes_skip_attention_model_0420_uint8.tflite", "wb").write(tflite_quant_model)

    elif j==5:
        #add opt
        print('')
        print('convert count= ', j+1)
        print('convert tflite opt model for coral tpu (uint8)')
        converter_5 = tf.lite.TFLiteConverter.from_keras_model(model)
        image_shape = (126, 64, 1)
        '''
        def representative_dataset_gen():
            num_calibration_steps = 1
            for _ in range(num_calibration_steps):
                image = tf.random.normal([1] + list(image_shape))
                yield [input]
        '''
        def representative_dataset_gen():
            num_calibration_images = 100
            for i in range(num_calibration_images):
        #         image = tf.random.normal([1] + list(image_shape))
                image = tf.random.uniform([1] + list(image_shape))
                yield [image]

        converter_5.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_5.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
        converter_5.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_5.inference_input_type = tf.uint8
        converter_5.inference_output_type = tf.uint8
        tflite_quant_model = converter_5.convert()
        open("freesound_umedia_20_classes_skip_attention_model_0420_opt_uint8.tflite", "wb").write(tflite_quant_model)
        
    elif j==6:
        print('')
        print('convert count= ', j+1)
        print('convert tflite model (float16)')
        converter_6 = tf.lite.TFLiteConverter.from_keras_model(model)
        #converter_6.target_spec.supported_types = [tf.lite.constants.FLOAT16]
        converter_6.target_spec.supported_types = [tf.float16]
        tflite_quant_model = converter_6.convert()
        open("freesound_umedia_20_classes_skip_attention_model_0420_fp16.tflite", "wb").write(tflite_quant_model)

    elif j==7:
        #add opt
        print('')
        print('convert count= ', j+1)
        print('convert tflite opt model (float16)')
        converter_7 = tf.lite.TFLiteConverter.from_keras_model(model)
        converter_7.optimizations = [tf.lite.Optimize.DEFAULT]
        #converter_7.target_spec.supported_types = [tf.lite.constants.FLOAT16]
        converter_7.target_spec.supported_types = [tf.float16]
        tflite_quant_model = converter_7.convert()
        open("freesound_umedia_20_classes_skip_attention_model_0420_opt_fp16.tflite", "wb").write(tflite_quant_model)
        
    elif j==8:
        print('')
        print('convert count= ', j+1)
        print('convert tflite model (float32 fallback)')
        converter_8 = tf.lite.TFLiteConverter.from_keras_model(model)
        image_shape = (126, 64, 1)
        def representative_dataset_gen():
            num_calibration_images = 100
            for i in range(num_calibration_images):
        #         image = tf.random.normal([1] + list(image_shape))
                image = tf.random.uniform([1] + list(image_shape))
                yield [image]

        converter_8.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
        tflite_quant_model = converter_8.convert()
        open("freesound_umedia_20_classes_skip_attention_model_0420_fp32fallback.tflite", "wb").write(tflite_quant_model)

    elif j==9:
        #add opt
        print('')
        print('convert count= ', j+1)
        print('convert tflite opt model (float32 fallback)')
        converter_9 = tf.lite.TFLiteConverter.from_keras_model(model)
        image_shape = (126, 64, 1)
        def representative_dataset_gen():
            num_calibration_images = 100
            for i in range(num_calibration_images):
        #         image = tf.random.normal([1] + list(image_shape))
                image = tf.random.uniform([1] + list(image_shape))
                yield [image]

        converter_9.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_9.representative_dataset = tf.lite.RepresentativeDataset(representative_dataset_gen)
        tflite_quant_model = converter_9.convert()
        open("freesound_umedia_20_classes_skip_attention_model_0420_opt_fp32fallback.tflite", "wb").write(tflite_quant_model)


