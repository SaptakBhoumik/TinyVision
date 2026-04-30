## 🟩 Latest Addition

I was planning to release the quantization experiments in V3 but I didnt get time + My exams started. But if I wait for my exams to end then it will be too late.

So I am releasing one of the countless experiments I have done on quantization. Kinda proud of it because it is a new quantization algo I made(Well I think it is new). But I admit it is not polished because I made a few mistakes which I realised later. But I am sharing it now because I think it is still interesting. It is 
quantile based(Simple yet robust). 

The test accuracy diffrence between QAT with my quantization and F32 model is around 5% which sounds bad but I think it is good because it is 3 bit quantization, I even quantized the activation to 3 bit(Actually everything that makes sense to quantize is quantized. So few parts are not quantized because it does not make sense to quantize them but most of it is quantized), every layer is quantized(well the ones that makes sense like CNN,Linear etc but not the ones that does not make sense like BatchNorm, ReLU etc), I even quantized the first and last layer and also the first input. Not to mention
this model is not that deep(Deep models benifit from QAT more) and also it is really compact with less than 60k parameters. So overall I think it is a good result

But I do admit the code is kinda shit because I was not focused on the code when I made it + a lot of things I have done that make 0 sense if you dont have all the files(They will make sense only if I upload all the file) but I still think it is interesting

Importants:-Proper control group study and comparision with existing methods is missing. Also the model on which I did QAT is an old version of the model in the sense, I have made a lot of new models which you can see in `cifar10_classifier/final/v1` directory that are much lighter but this QAT study was done before it when my models were not that effecient

Check out `cifar10_classifier/proto` if you are interested