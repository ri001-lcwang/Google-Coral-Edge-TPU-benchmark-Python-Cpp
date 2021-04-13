
#Note: To guarantee running without 
#"RuntimeError: Internal: Unsupported data type in custom op handler: 24348008Node number 2 (EdgeTpuDelegateForCustomOp) failed to prepare.",
#it is necessary to follow up "install software package" procedure first.

## install software package
Follow up https://github.com/Namburger/edgetpu-minimal-example



## Compile
cd build

## Build Instructions
sudo rm -r *
cmake ..
make



## Run
../out/demo


## Note: It is implemented with Auto detect Edgetpu device.
# Result while TPU is connected.
pi@raspberrypi:~/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build $ ../out/demo
Note: Detect edgetpu devices is existence. Run TPU & CPU.

  Benchmark result:
  Device | Opt model | Data type | Pred_threads | Accuracy(%) | Load model(s) | 1st pred(s)  | Avg 17 pred(s) | Avg 17 set_io + pred(s) | Pthread
   TPU   |     N     |    int8   |       1      |   70.59     |    0.06046    |    0.02232   |     0.00883    |       0.00927           |    N
   TPU   |     N     |    int8   |       2      |   70.59     |    0.06115    |    0.01421   |     0.00780    |       0.00826           |    N
   TPU   |     N     |    int8   |       3      |   70.59     |    0.06364    |    0.00952   |     0.00817    |       0.00862           |    N
   TPU   |     N     |    int8   |       4      |   70.59     |    0.06251    |    0.00853   |     0.00841    |       0.00886           |    N
   TPU   |     N     |    int8   |       1      |   70.59     |    0.06322    |    0.01581   |     0.00855    |       0.00900           |    Y
   TPU   |     N     |    int8   |       2      |   70.59     |    0.06580    |    0.01656   |     0.00810    |       0.00854           |    Y
   TPU   |     N     |    int8   |       3      |   70.59     |    0.06394    |    0.01624   |     0.00873    |       0.00917           |    Y
   TPU   |     N     |    int8   |       4      |   70.59     |    0.06531    |    0.01639   |     0.00874    |       0.00919           |    Y
   TPU   |     Y     |    int8   |       1      |   76.47     |    0.06559    |    0.01656   |     0.00868    |       0.00913           |    N
   TPU   |     Y     |    int8   |       2      |   76.47     |    0.06563    |    0.00868   |     0.00832    |       0.00877           |    N
   TPU   |     Y     |    int8   |       3      |   76.47     |    0.06383    |    0.00838   |     0.00826    |       0.00872           |    N
   TPU   |     Y     |    int8   |       4      |   76.47     |    0.06512    |    0.00847   |     0.00827    |       0.00872           |    N
   TPU   |     Y     |    int8   |       1      |   76.47     |    0.06342    |    0.01678   |     0.00879    |       0.00925           |    Y
   TPU   |     Y     |    int8   |       2      |   76.47     |    0.06380    |    0.01665   |     0.00777    |       0.00822           |    Y
   TPU   |     Y     |    int8   |       3      |   76.47     |    0.06483    |    0.01651   |     0.00881    |       0.00926           |    Y
   TPU   |     Y     |    int8   |       4      |   76.47     |    0.06375    |    0.01575   |     0.00782    |       0.00827           |    Y
   TPU   |     N     |   uint8   |       1      |   64.71     |    0.06549    |    0.01353   |     0.00720    |       0.00765           |    N
   TPU   |     N     |   uint8   |       2      |   64.71     |    0.06312    |    0.00684   |     0.00673    |       0.00717           |    N
   TPU   |     N     |   uint8   |       3      |   64.71     |    0.06243    |    0.00680   |     0.00671    |       0.00716           |    N
   TPU   |     N     |   uint8   |       4      |   64.71     |    0.06292    |    0.00683   |     0.00681    |       0.00725           |    N
   TPU   |     N     |   uint8   |       1      |   64.71     |    0.06324    |    0.01358   |     0.00787    |       0.00830           |    Y
   TPU   |     N     |   uint8   |       2      |   64.71     |    0.06296    |    0.01651   |     0.00806    |       0.00848           |    Y
   TPU   |     N     |   uint8   |       3      |   64.71     |    0.06284    |    0.01332   |     0.00741    |       0.00784           |    Y
   TPU   |     N     |   uint8   |       4      |   64.71     |    0.06378    |    0.01348   |     0.00787    |       0.00831           |    Y
   TPU   |     Y     |   uint8   |       1      |   82.35     |    0.06364    |    0.01559   |     0.00834    |       0.00878           |    N
   TPU   |     Y     |   uint8   |       2      |   82.35     |    0.06430    |    0.00681   |     0.00755    |       0.00798           |    N
   TPU   |     Y     |   uint8   |       3      |   82.35     |    0.06304    |    0.00840   |     0.00819    |       0.00862           |    N
   TPU   |     Y     |   uint8   |       4      |   82.35     |    0.06296    |    0.00822   |     0.00809    |       0.00851           |    N
   TPU   |     Y     |   uint8   |       1      |   82.35     |    0.06353    |    0.01629   |     0.00856    |       0.00900           |    Y
   TPU   |     Y     |   uint8   |       2      |   82.35     |    0.06325    |    0.01635   |     0.00871    |       0.00915           |    Y
   TPU   |     Y     |   uint8   |       3      |   82.35     |    0.06285    |    0.01654   |     0.00871    |       0.00915           |    Y
   TPU   |     Y     |   uint8   |       4      |   82.35     |    0.06314    |    0.01638   |     0.00872    |       0.00915           |    Y
   CPU   |     N     |    int8   |       1      |   76.47     |    0.06242    |    0.30539   |     0.29384    |       0.29428           |    N
   CPU   |     N     |    int8   |       2      |   76.47     |    0.06061    |    0.31644   |     0.30675    |       0.30778           |    N
   CPU   |     N     |    int8   |       3      |   76.47     |    0.05996    |    0.32980   |     0.33069    |       0.33114           |    N
   CPU   |     N     |    int8   |       4      |   76.47     |    0.06152    |    0.37010   |     0.35296    |       0.35341           |    N
   CPU   |     N     |    int8   |       1      |   76.47     |    0.06310    |    0.30711   |     0.29624    |       0.29668           |    Y
   CPU   |     N     |    int8   |       2      |   76.47     |    0.05890    |    0.31694   |     0.30520    |       0.30564           |    Y
   CPU   |     N     |    int8   |       3      |   76.47     |    0.05871    |    0.32810   |     0.32367    |       0.32413           |    Y
   CPU   |     N     |    int8   |       4      |   76.47     |    0.06214    |    0.36738   |     0.35124    |       0.35169           |    Y
   CPU   |     Y     |    int8   |       1      |   76.47     |    0.06037    |    0.30579   |     0.29499    |       0.29543           |    N
   CPU   |     Y     |    int8   |       2      |   76.47     |    0.06050    |    0.32289   |     0.30759    |       0.30804           |    N
   CPU   |     Y     |    int8   |       3      |   76.47     |    0.06135    |    0.34587   |     0.33206    |       0.33251           |    N
   CPU   |     Y     |    int8   |       4      |   76.47     |    0.05853    |    0.35905   |     0.35064    |       0.35109           |    N
   CPU   |     Y     |    int8   |       1      |   76.47     |    0.05954    |    0.30813   |     0.29701    |       0.29745           |    Y
   CPU   |     Y     |    int8   |       2      |   76.47     |    0.05964    |    0.31942   |     0.30771    |       0.30815           |    Y
   CPU   |     Y     |    int8   |       3      |   76.47     |    0.06001    |    0.35045   |     0.33077    |       0.33122           |    Y
   CPU   |     Y     |    int8   |       4      |   76.47     |    0.06132    |    0.37001   |     0.35710    |       0.35756           |    Y
   CPU   |     N     |   uint8   |       1      |   82.35     |    0.06015    |    0.30837   |     0.29656    |       0.29699           |    N
   CPU   |     N     |   uint8   |       2      |   82.35     |    0.05908    |    0.31527   |     0.30275    |       0.30318           |    N
   CPU   |     N     |   uint8   |       3      |   82.35     |    0.05891    |    0.33355   |     0.32634    |       0.32678           |    N
   CPU   |     N     |   uint8   |       4      |   82.35     |    0.05963    |    0.35296   |     0.35163    |       0.35207           |    N
   CPU   |     N     |   uint8   |       1      |   82.35     |    0.05899    |    0.31078   |     0.29984    |       0.30028           |    Y
   CPU   |     N     |   uint8   |       2      |   82.35     |    0.06088    |    0.31364   |     0.30403    |       0.30445           |    Y
   CPU   |     N     |   uint8   |       3      |   82.35     |    0.06166    |    0.34848   |     0.33164    |       0.33207           |    Y
   CPU   |     N     |   uint8   |       4      |   82.35     |    0.06063    |    0.36229   |     0.35890    |       0.35933           |    Y
   CPU   |     Y     |   uint8   |       1      |   76.47     |    0.05874    |    0.30718   |     0.29612    |       0.29655           |    N
   CPU   |     Y     |   uint8   |       2      |   76.47     |    0.05905    |    0.31806   |     0.30668    |       0.30711           |    N
   CPU   |     Y     |   uint8   |       3      |   76.47     |    0.06017    |    0.33762   |     0.32991    |       0.33035           |    N
   CPU   |     Y     |   uint8   |       4      |   76.47     |    0.05855    |    0.35936   |     0.35344    |       0.35387           |    N
   CPU   |     Y     |   uint8   |       1      |   76.47     |    0.05852    |    0.30976   |     0.29891    |       0.29934           |    Y
   CPU   |     Y     |   uint8   |       2      |   76.47     |    0.05967    |    0.31970   |     0.30874    |       0.30916           |    Y
   CPU   |     Y     |   uint8   |       3      |   76.47     |    0.05987    |    0.34972   |     0.32576    |       0.32738           |    Y
   CPU   |     Y     |   uint8   |       4      |   76.47     |    0.05960    |    0.35348   |     0.35218    |       0.35261           |    Y
   CPU   |     N     |     fp16  |       1      |   82.35     |    0.22390    |    0.89081   |     0.85520    |       0.85537           |    N
   CPU   |     N     |     fp16  |       2      |   82.35     |    0.25109    |    0.92613   |     0.85423    |       0.85440           |    N
   CPU   |     N     |     fp16  |       3      |   82.35     |    0.22764    |    0.89095   |     0.84804    |       0.84821           |    N
   CPU   |     N     |     fp16  |       4      |   82.35     |    0.22505    |    0.89162   |     0.84688    |       0.84705           |    N
   CPU   |     N     |     fp16  |       1      |   82.35     |    0.22239    |    0.89140   |     0.86139    |       0.86156           |    Y
   CPU   |     N     |     fp16  |       2      |   82.35     |    0.22891    |    0.88842   |     0.84452    |       0.84469           |    Y
   CPU   |     N     |     fp16  |       3      |   82.35     |    0.22743    |    0.90156   |     0.84822    |       0.84839           |    Y
   CPU   |     N     |     fp16  |       4      |   82.35     |    0.22226    |    0.89118   |     0.84627    |       0.84644           |    Y
   CPU   |     Y     |     fp16  |       1      |   82.35     |    0.12105    |    1.05430   |     0.84850    |       0.84867           |    N
   CPU   |     Y     |     fp16  |       2      |   82.35     |    0.12372    |    1.05679   |     0.84930    |       0.84946           |    N
   CPU   |     Y     |     fp16  |       3      |   82.35     |    0.12395    |    1.05248   |     0.84921    |       0.84938           |    N
   CPU   |     Y     |     fp16  |       4      |   82.35     |    0.12155    |    1.05859   |     0.84842    |       0.84858           |    N
   CPU   |     Y     |     fp16  |       1      |   82.35     |    0.11976    |    1.05469   |     0.85539    |       0.85556           |    Y
   CPU   |     Y     |     fp16  |       2      |   82.35     |    0.12294    |    1.05594   |     0.84780    |       0.84796           |    Y
   CPU   |     Y     |     fp16  |       3      |   82.35     |    0.12295    |    1.06439   |     0.85030    |       0.85046           |    Y
   CPU   |     Y     |     fp16  |       4      |   82.35     |    0.12005    |    1.05804   |     0.85071    |       0.85088           |    Y
   CPU   |     N     |     fp32fb|       1      |   82.35     |    0.25234    |    0.93135   |     0.86319    |       0.86337           |    N
   CPU   |     N     |     fp32fb|       2      |   82.35     |    0.22493    |    0.89339   |     0.85061    |       0.85078           |    N
   CPU   |     N     |     fp32fb|       3      |   82.35     |    0.22766    |    0.89079   |     0.84872    |       0.84889           |    N
   CPU   |     N     |     fp32fb|       4      |   82.35     |    0.22347    |    0.89092   |     0.85554    |       0.85571           |    N
   CPU   |     N     |     fp32fb|       1      |   82.35     |    0.24623    |    0.93250   |     0.85515    |       0.85532           |    Y
   CPU   |     N     |     fp32fb|       2      |   82.35     |    0.22549    |    0.88892   |     0.84671    |       0.84688           |    Y
   CPU   |     N     |     fp32fb|       3      |   82.35     |    0.22652    |    0.89305   |     0.84699    |       0.84716           |    Y
   CPU   |     N     |     fp32fb|       4      |   82.35     |    0.22417    |    0.88513   |     0.86512    |       0.86529           |    Y
   CPU   |     Y     |     fp32fb|       1      |   76.47     |    0.06100    |    0.30606   |     0.29504    |       0.29521           |    N
   CPU   |     Y     |     fp32fb|       2      |   76.47     |    0.05838    |    0.31968   |     0.30655    |       0.30672           |    N
   CPU   |     Y     |     fp32fb|       3      |   76.47     |    0.06158    |    0.33740   |     0.32632    |       0.32649           |    N
   CPU   |     Y     |     fp32fb|       4      |   76.47     |    0.06030    |    0.35132   |     0.35359    |       0.35377           |    N
   CPU   |     Y     |     fp32fb|       1      |   76.47     |    0.05861    |    0.30958   |     0.29706    |       0.29723           |    Y
   CPU   |     Y     |     fp32fb|       2      |   76.47     |    0.06272    |    0.31769   |     0.30503    |       0.30520           |    Y
   CPU   |     Y     |     fp32fb|       3      |   76.47     |    0.05950    |    0.32686   |     0.32208    |       0.32226           |    Y
   CPU   |     Y     |     fp32fb|       4      |   76.47     |    0.05661    |    0.36596   |     0.35023    |       0.35040           |    Y
   CPU   |     N     |     fp32  |       1      |   82.35     |    0.22144    |    0.88773   |     0.84372    |       0.84389           |    N
   CPU   |     N     |     fp32  |       2      |   82.35     |    0.22248    |    0.88864   |     0.86020    |       0.86037           |    N
   CPU   |     N     |     fp32  |       3      |   82.35     |    0.22912    |    0.88911   |     0.84612    |       0.84629           |    N
   CPU   |     N     |     fp32  |       4      |   82.35     |    0.22609    |    0.88858   |     0.84622    |       0.84639           |    N
   CPU   |     N     |     fp32  |       1      |   82.35     |    0.22412    |    0.89262   |     0.84774    |       0.84791           |    Y
   CPU   |     N     |     fp32  |       2      |   82.35     |    0.22342    |    0.91976   |     0.86734    |       0.86751           |    Y
   CPU   |     N     |     fp32  |       3      |   82.35     |    0.22525    |    0.88646   |     0.84752    |       0.84769           |    Y
   CPU   |     N     |     fp32  |       4      |   82.35     |    0.22415    |    0.88623   |     0.84474    |       0.84491           |    Y
   CPU   |     Y     |     fp32  |       1      |   82.35     |    0.06131    |    0.49183   |     0.47333    |       0.47350           |    N
   CPU   |     Y     |     fp32  |       2      |   82.35     |    0.05806    |    0.49099   |     0.50559    |       0.50577           |    N
   CPU   |     Y     |     fp32  |       3      |   82.35     |    0.07148    |    0.55374   |     0.48280    |       0.48297           |    N
   CPU   |     Y     |     fp32  |       4      |   82.35     |    0.05818    |    0.49291   |     0.47440    |       0.47457           |    N
   CPU   |     Y     |     fp32  |       1      |   82.35     |    0.05883    |    0.49502   |     0.47660    |       0.47677           |    Y
   CPU   |     Y     |     fp32  |       2      |   82.35     |    0.05810    |    0.49298   |     0.47527    |       0.47544           |    Y
   CPU   |     Y     |     fp32  |       3      |   82.35     |    0.05826    |    0.49142   |     0.47314    |       0.47331           |    Y
   CPU   |     Y     |     fp32  |       4      |   82.35     |    0.05879    |    0.49160   |     0.48536    |       0.48553           |    Y
# => Analysis.
# => Best result.
#  TPU   |     Y     |   uint8   |       2      |   82.35     |    0.06430    |    0.00681   |     0.00755    |       0.00798           |    N
#  CPU   |     N     |   uint8   |       1      |   82.35     |    0.06015    |    0.30837   |     0.29656    |       0.29699           |    N
# => Speedup for Avg 17 set_io + pred = 37.2
#  CPU   |     Y     |     fp32  |       3      |   82.35     |    0.05826    |    0.49142   |     0.47314    |       0.47331           |    Y 
# => Speedup for Avg 17 set_io + pred = 59.3



# Result while TPU is not connected.
pi@raspberrypi:~/Desktop/hawkeye/stanley_test_files/test_edgetpu_minimal_example/build $ ../out/demo
Warning: Detect edgetpu devices is none. Run without TPU.

  Benchmark result:
  Device | Opt model | Data type | Pred_threads | Accuracy(%) | Load model(s) | 1st pred(s)  | Avg 17 pred(s) | Avg 17 set_io + pred(s) | Pthread
   CPU   |     N     |    int8   |       1      |   76.47     |    0.25423    |    0.33603   |     0.31402    |       0.31445           |    N
   CPU   |     N     |    int8   |       2      |   76.47     |    0.06084    |    0.34173   |     0.30726    |       0.30769           |    N
   CPU   |     N     |    int8   |       3      |   76.47     |    0.05895    |    0.37720   |     0.33298    |       0.33341           |    N
   CPU   |     N     |    int8   |       4      |   76.47     |    0.05761    |    0.40669   |     0.36103    |       0.36147           |    N
   CPU   |     N     |    int8   |       1      |   76.47     |    0.05600    |    0.31520   |     0.29876    |       0.29919           |    Y
   CPU   |     N     |    int8   |       2      |   76.47     |    0.05846    |    0.33426   |     0.30901    |       0.30944           |    Y
   CPU   |     N     |    int8   |       3      |   76.47     |    0.05924    |    0.34891   |     0.32848    |       0.32891           |    Y
   CPU   |     N     |    int8   |       4      |   76.47     |    0.05944    |    0.41989   |     0.36724    |       0.36772           |    Y
   CPU   |     Y     |    int8   |       1      |   76.47     |    0.06270    |    0.31228   |     0.29958    |       0.30001           |    N
   CPU   |     Y     |    int8   |       2      |   76.47     |    0.05877    |    0.34570   |     0.30814    |       0.30857           |    N
   CPU   |     Y     |    int8   |       3      |   76.47     |    0.05859    |    0.35992   |     0.32813    |       0.32856           |    N
   CPU   |     Y     |    int8   |       4      |   76.47     |    0.05786    |    0.40845   |     0.35860    |       0.35904           |    N
   CPU   |     Y     |    int8   |       1      |   76.47     |    0.05884    |    0.31723   |     0.29449    |       0.29492           |    Y
   CPU   |     Y     |    int8   |       2      |   76.47     |    0.05772    |    0.32594   |     0.32641    |       0.32685           |    Y
   CPU   |     Y     |    int8   |       3      |   76.47     |    0.06651    |    0.38468   |     0.35170    |       0.35214           |    Y
   CPU   |     Y     |    int8   |       4      |   76.47     |    0.06748    |    0.52461   |     0.37167    |       0.37211           |    Y
   CPU   |     N     |   uint8   |       1      |   82.35     |    0.05959    |    0.32029   |     0.29864    |       0.29905           |    N
   CPU   |     N     |   uint8   |       2      |   82.35     |    0.05915    |    0.33571   |     0.30970    |       0.31011           |    N
   CPU   |     N     |   uint8   |       3      |   82.35     |    0.05979    |    0.35257   |     0.33268    |       0.33310           |    N
   CPU   |     N     |   uint8   |       4      |   82.35     |    0.05806    |    0.41587   |     0.36323    |       0.36384           |    N
   CPU   |     N     |   uint8   |       1      |   82.35     |    0.05812    |    0.31427   |     0.29442    |       0.29483           |    Y
   CPU   |     N     |   uint8   |       2      |   82.35     |    0.05934    |    0.34407   |     0.30722    |       0.30764           |    Y
   CPU   |     N     |   uint8   |       3      |   82.35     |    0.05904    |    0.36279   |     0.33156    |       0.33198           |    Y
   CPU   |     N     |   uint8   |       4      |   82.35     |    0.05912    |    0.41126   |     0.35521    |       0.35564           |    Y
   CPU   |     Y     |   uint8   |       1      |   76.47     |    0.05937    |    0.31735   |     0.29927    |       0.29968           |    N
   CPU   |     Y     |   uint8   |       2      |   76.47     |    0.05919    |    0.33723   |     0.30735    |       0.30776           |    N
   CPU   |     Y     |   uint8   |       3      |   76.47     |    0.05787    |    0.37061   |     0.33122    |       0.33164           |    N
   CPU   |     Y     |   uint8   |       4      |   76.47     |    0.05769    |    0.41682   |     0.39479    |       0.39521           |    N
   CPU   |     Y     |   uint8   |       1      |   76.47     |    0.06505    |    0.34078   |     0.31022    |       0.31064           |    Y
   CPU   |     Y     |   uint8   |       2      |   76.47     |    0.06032    |    0.33117   |     0.30926    |       0.30968           |    Y
   CPU   |     Y     |   uint8   |       3      |   76.47     |    0.05943    |    0.36301   |     0.32857    |       0.32899           |    Y
   CPU   |     Y     |   uint8   |       4      |   76.47     |    0.05925    |    0.41611   |     0.35870    |       0.35912           |    Y
   CPU   |     N     |     fp16  |       1      |   82.35     |    0.23966    |    0.89163   |     0.84250    |       0.84266           |    N
   CPU   |     N     |     fp16  |       2      |   82.35     |    0.24115    |    0.89735   |     0.84531    |       0.84546           |    N
   CPU   |     N     |     fp16  |       3      |   82.35     |    0.24174    |    0.88906   |     0.85939    |       0.85955           |    N
   CPU   |     N     |     fp16  |       4      |   82.35     |    0.26582    |    0.93523   |     0.84599    |       0.84615           |    N
   CPU   |     N     |     fp16  |       1      |   82.35     |    0.23981    |    0.88278   |     0.84191    |       0.84207           |    Y
   CPU   |     N     |     fp16  |       2      |   82.35     |    0.24122    |    0.89225   |     0.84510    |       0.84526           |    Y
   CPU   |     N     |     fp16  |       3      |   82.35     |    0.24106    |    0.88531   |     0.86313    |       0.86329           |    Y
   CPU   |     N     |     fp16  |       4      |   82.35     |    0.24053    |    0.88721   |     0.84120    |       0.84135           |    Y
   CPU   |     Y     |     fp16  |       1      |   82.35     |    0.12204    |    1.05190   |     0.84599    |       0.84614           |    N
   CPU   |     Y     |     fp16  |       2      |   82.35     |    0.12121    |    1.05852   |     0.84709    |       0.84724           |    N
   CPU   |     Y     |     fp16  |       3      |   82.35     |    0.12309    |    1.05914   |     0.86764    |       0.86780           |    N
   CPU   |     Y     |     fp16  |       4      |   82.35     |    0.12135    |    1.06310   |     0.84722    |       0.84737           |    N
   CPU   |     Y     |     fp16  |       1      |   82.35     |    0.12282    |    1.05866   |     0.84447    |       0.84462           |    Y
   CPU   |     Y     |     fp16  |       2      |   82.35     |    0.12005    |    1.05235   |     0.84447    |       0.84462           |    Y
   CPU   |     Y     |     fp16  |       3      |   82.35     |    0.11931    |    1.06954   |     0.86064    |       0.86079           |    Y
   CPU   |     Y     |     fp16  |       4      |   82.35     |    0.12160    |    1.05654   |     0.84324    |       0.84340           |    Y
   CPU   |     N     |     fp32fb|       1      |   82.35     |    0.23942    |    0.88841   |     0.84268    |       0.84284           |    N
   CPU   |     N     |     fp32fb|       2      |   82.35     |    0.24133    |    0.89411   |     0.84785    |       0.84801           |    N
   CPU   |     N     |     fp32fb|       3      |   82.35     |    0.26555    |    0.93626   |     0.85527    |       0.85543           |    N
   CPU   |     N     |     fp32fb|       4      |   82.35     |    0.24324    |    0.89403   |     0.84414    |       0.84429           |    N
   CPU   |     N     |     fp32fb|       1      |   82.35     |    0.24202    |    0.89233   |     0.84490    |       0.84506           |    Y
   CPU   |     N     |     fp32fb|       2      |   82.35     |    0.23849    |    0.88498   |     0.85524    |       0.85539           |    Y
   CPU   |     N     |     fp32fb|       3      |   82.35     |    0.27110    |    0.93215   |     0.84493    |       0.84508           |    Y
   CPU   |     N     |     fp32fb|       4      |   82.35     |    0.24068    |    0.89309   |     0.84309    |       0.84325           |    Y
   CPU   |     Y     |     fp32fb|       1      |   76.47     |    0.05962    |    0.31782   |     0.29915    |       0.29931           |    N
   CPU   |     Y     |     fp32fb|       2      |   76.47     |    0.05891    |    0.33872   |     0.30913    |       0.30928           |    N
   CPU   |     Y     |     fp32fb|       3      |   76.47     |    0.05876    |    0.35615   |     0.32743    |       0.32759           |    N
   CPU   |     Y     |     fp32fb|       4      |   76.47     |    0.05880    |    0.41112   |     0.36072    |       0.36088           |    N
   CPU   |     Y     |     fp32fb|       1      |   76.47     |    0.05786    |    0.31820   |     0.29884    |       0.29900           |    Y
   CPU   |     Y     |     fp32fb|       2      |   76.47     |    0.05904    |    0.33121   |     0.33257    |       0.33274           |    Y
   CPU   |     Y     |     fp32fb|       3      |   76.47     |    0.06733    |    0.40599   |     0.35601    |       0.35617           |    Y
   CPU   |     Y     |     fp32fb|       4      |   76.47     |    0.06659    |    0.40634   |     0.35752    |       0.35768           |    Y
   CPU   |     N     |     fp32  |       1      |   82.35     |    0.23743    |    0.89464   |     0.84331    |       0.84347           |    N
   CPU   |     N     |     fp32  |       2      |   82.35     |    0.24122    |    0.88962   |     0.84524    |       0.84540           |    N
   CPU   |     N     |     fp32  |       3      |   82.35     |    0.23837    |    0.89159   |     0.84212    |       0.84227           |    N
   CPU   |     N     |     fp32  |       4      |   82.35     |    0.23876    |    0.88778   |     0.86148    |       0.86164           |    N
   CPU   |     N     |     fp32  |       1      |   82.35     |    0.23826    |    0.89797   |     0.84466    |       0.84481           |    Y
   CPU   |     N     |     fp32  |       2      |   82.35     |    0.24205    |    0.89540   |     0.84251    |       0.84267           |    Y
   CPU   |     N     |     fp32  |       3      |   82.35     |    0.24093    |    0.89812   |     0.84829    |       0.84845           |    Y
   CPU   |     N     |     fp32  |       4      |   82.35     |    0.26671    |    0.93014   |     0.85946    |       0.85962           |    Y
   CPU   |     Y     |     fp32  |       1      |   82.35     |    0.05962    |    0.50366   |     0.47684    |       0.47699           |    N
   CPU   |     Y     |     fp32  |       2      |   82.35     |    0.05854    |    0.51588   |     0.47498    |       0.47513           |    N
   CPU   |     Y     |     fp32  |       3      |   82.35     |    0.05841    |    0.51427   |     0.47591    |       0.47607           |    N
   CPU   |     Y     |     fp32  |       4      |   82.35     |    0.05786    |    0.51890   |     0.47181    |       0.47197           |    N
   CPU   |     Y     |     fp32  |       1      |   82.35     |    0.05784    |    0.50406   |     0.50105    |       0.50121           |    Y
   CPU   |     Y     |     fp32  |       2      |   82.35     |    0.06397    |    0.58594   |     0.49899    |       0.49915           |    Y
   CPU   |     Y     |     fp32  |       3      |   82.35     |    0.05900    |    0.49873   |     0.47438    |       0.47454           |    Y
   CPU   |     Y     |     fp32  |       4      |   82.35     |    0.05896    |    0.51037   |     0.47396    |       0.47412           |    Y

