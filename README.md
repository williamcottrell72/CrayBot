# PROJECT CRABYOT

This repo provides code for a general purpose chatbot.  The core code is contained in **craybot.py** where the class CRAYBOT is constructed.  The CRAYBOT class workds just like an sklearn module, i.e., there are **fit** and **predict** methods that the user can call after instantiating an instance of the class.  The code should be used as follows:

```python
import craybot

# First, choose a method for converting words to vectors.  The method will be fed to CRAYBOT using the variable `encoder'.  There are many options and, in principle this step is relatively independent of the later training steps.  In the future, I plan to automate this choice and include the encoding as part of the training algorithm.   For the purposes of this example, I'll just choose an encoder that is designed to mesh well with the target text - Alice in Wonderland.

with open('text_files/carroll-alice.txt','r') as f:
    alice=f.readlines()

# Next, we call the 'clean_corpus' preprocessing method from craybot to feed the string `alice' into the vectorizer.  The output is simply a long python string.

string=' '.join(craybot.clean_corpus(aiw))
w2v=gensim.models.Word2Vec([string.split()], size=100, window=5, min_count=1, workers=4,sg=1)

# Now we instantiate an instance of the bot using the vectorizer above.  


wonder_bot=craybot.CRAYBOT(temperature = 1,sent_length=3,n_samples=100,hidden_size=200,n_layers=4,max_iter=10,encoder=w2v)

# As with scikit learn, we have a fit method.  In principle, we could use a text other than 'Alice in Wonderland'.

wonder_bot.fit(string)

# Now the bot is 'fit' and can be used to generate text given a string as a seed.  For instance:

wonder_bot.predict(['He said that'],10)
```
OUTPUT: "He said that grin spoon . offended , cup 'he fixed 'talking head"

Obviously this could be a bit better.  The class also stores losses for each iteration in **wonder_bot.losses** so that diagnostics can be performed.
