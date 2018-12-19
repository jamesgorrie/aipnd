# AI Programming with Python Nanodegree Program

A lot of inspiration and knowledge has been taken
from [Pytorche's great tutorials](https://pytorch.org/tutorials/),
specifically the [tutorial on transfer learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## Pain points
The programming in and of itself is actually quite straight forward,
but I was receiving really low accuracy results from the training.

It turned out my classifier was poorly setup, and after talking to people
over Slack and scouring the internet this seemed to work:
```python
classifier = nn.Sequential(OrderedDict([
            ('dropout1', nn.Dropout(dropout)),
            ('fc1', nn.Linear(num_filters, hidden_units)),
            ('relu1', nn.ReLU(True)),
            ('dropout2', nn.Dropout(dropout)),
            ('fc2', nn.Linear(hidden_units, hidden_units)),
            ('relu2', nn.ReLU(True)),
            ('fc3', nn.Linear(hidden_units, num_labels)),
            ]))
```

I would like to try get a greater intuition as to what classifiers would
work best for which models, and to learn about how different optimizers,
criterion and schedulers might help produce even more accurate results.

Graphing from tensors is also something I haven't found massively intuitive,
and need to spend more time getting right.

## Learning notes
* Learning in the morning was a lot more productive,
  especially with full-time work, which can wear your brain out before the evening.
* Having done it, in the future I would run through the course quickly to give
  an indication of where we would be heading, and get an overview of the feel for the course,
  and then run through it more thoroughly. This would allow me to at least have a feeling for 
  which parts of the course are worth spending time on, as I spent a huge amount of time going
  through the math side (which was great!) - but the focus felt like it was on the use of that math.
