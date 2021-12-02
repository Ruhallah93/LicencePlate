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


<h1>How to use DataMaker2 for YOLO</h1>
<h3>Example</h3>

In order to generate 20000 data using 10 threads for yolo, run this command:

```
python3 DataMaker2.py --size 20000 --workers 10 --model yolo
```

<h2 style="color: orange;"> Caution </h2>

Remeber to create a directory named ``` output ``` at the same directory as ```DataMaker.py```


<h1>How to use DataMaker2 for UNET</h1>
<h3>Example</h3>

In order to generate 5000 train data using 10 threads for unet, run this command:

```
python3 DataMaker2.py --size 5000 --workers 10 --model unet --type train
```

In order to generate 1000 test data using 10 threads for unet, run this command:

```
python3 DataMaker2.py --size 1000 --workers 10 --model unet --type test
```

<h2 style="color: orange;"> Caution </h2>

Remeber to create a directory named ``` output ``` at the same directory as ```DataMaker.py```, then create a directory named ```unetData```, put two directories with names ```train``` and ```test``` and each one put two directories named ```images``` and ```masks```.