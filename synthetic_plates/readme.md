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


<h1>How to use DataMaker2</h1>
<h3>Example</h3>

In order to generate 20000 data using 10 threads for yolo, run this command:

```
python3 DataMaker2.py --size 20000 --workers 10
```

<h2> Caution </h2>

Remeber to create a directory named ``` output ``` at the same directory as ```DataMaker.py```