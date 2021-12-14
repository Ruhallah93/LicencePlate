<h3>
HBD BIDEH
2021 Dec 2
</h3>

Some Generated Examples:<br/>
<img src="generated_examples/1.png">
<img src="generated_examples/2.png"><br/>
<img src="generated_examples/3.png">
<img src="generated_examples/4.png"><br/>
<img src="generated_examples/5.png">

Generated Noisy Examples:<br/>
<img src="generated_examples/noisy1.png">
<img src="generated_examples/noisy2.png"><br/>
<img src="generated_examples/noisy3.png">
<img src="generated_examples/noisy4.png"><br/>
<img src="generated_examples/noisy5.png">


<h1>How to use DataMaker for YOLO</h1>
<h3>Example</h3>

In order to generate 20000 data using 10 threads for yolo, run this command:

```
python3 DataMaker.py --size 20000 --workers 10 --img_size 500 400 --save_bounding_boxes --mask_state "grayscale" --address "output/yolo/train"

```
Or:

```
python3 DataMaker.py --size 20000 --workers 10 --img_size 500 400 --save_bounding_boxes --mask_state "grayscale" --address "output/yolo/test"

```


<h1>How to use DataMaker for UNET, CycleGAN, ...</h1>
<h3>Example</h3>

In order to generate 5000 data using 10 threads for unet, run this command:

```
python3 DataMaker.py --size 5000 --workers 10 --img_size 500 400 --save_mask --mask_state "colorful" --address "output/unet/train"
```

Or:

```
python3 DataMaker.py --size 5000 --workers 10 --img_size 500 400 --save_mask --mask_state "colorful" --address "output/unet/test"
```