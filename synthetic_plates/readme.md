<h3>
HBD BIDEH
2021 Dec 2
</h3>


Generated Examples and Masks:<br/>
<img src="generated_examples/01.png">
<img src="generated_examples/01m.png"><br/>
<img src="generated_examples/02.png">
<img src="generated_examples/02m.png"><br/>
<img src="generated_examples/03.png">
<img src="generated_examples/03m.png"><br/>
<img src="generated_examples/04.png">
<img src="generated_examples/04m.png"><br/>
<img src="generated_examples/05.png">
<img src="generated_examples/05m.png"><br/>

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