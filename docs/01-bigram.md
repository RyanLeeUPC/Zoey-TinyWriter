# Chapter 1: The bigram model

> **Goal:** build the smallest thing that deserves to be called a language model, train it, and watch it go from random noise to something that *looks* like English.
>
> **Code:** [`zoey/models/bigram.py`](../zoey/models/bigram.py) · **Config:** [`configs/01-bigram.toml`](../configs/01-bigram.toml) · **Viewer:** Chapter 1

## The one job of a language model

Every language model, from this one to the largest chatbot, does exactly one thing:

> Given some text, predict what comes next.

That's it. It doesn't "write a story." It answers *"what's the next token?"* over and over, and we append each answer to the text. Writing is just prediction on a loop.

In this chapter a **token** is a single character. (Chapter 2 upgrades that.)

## The simplest possible guesser

What's the simplest way to predict the next character? Look at the current character and ask: *"in all the text I've seen, what usually comes after this?"*

- After `q`, it's almost always `u`.
- After `.`, it's usually a space.
- After `t`, it's often `h` (as in *the*, *that*, *then*).

A model that predicts from **one** previous character is called a **bigram** model ("bi" = two: the current character and the next). Its entire knowledge fits in one table:

```
                    next character →
                    a      b      c     ...    u    ...
current      a   [ 0.01   0.03   0.04  ...                ]
character    b   [ 0.09   0.02   0.00  ...                ]
             q   [ 0.00   0.00   0.00  ...  0.97  ...     ]
```

Each row is a probability distribution: the numbers add up to 1 (100%).

We have 97 characters, so the table is 97 × 97 = **9,409 numbers**. Those are all of TinyWriter's **parameters** in this chapter. GPT-style models have millions to billions of parameters, but they get trained in exactly the same way.

## Learning the table instead of counting

You could fill in this table by counting pairs of letters in the training text. That would work! But we're going to do something that looks like overkill: start the table **full of random numbers** and improve it with **gradient descent**.

Why? Because counting only works for this one tiny model. Gradient descent works for *every* model in this course, including the full transformer in chapter 5. The bigram model is small enough to see every single parameter change while it learns.

### The model, in code

```python
class BigramModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The whole model: one row of scores per character.
        self.table = nn.Embedding(config.vocab_size, config.vocab_size)

    def forward(self, idx, targets=None):
        logits = self.table(idx)               # look up each character's row
        loss = F.cross_entropy(...)            # how surprised were we?
        return logits, loss
```

Two ideas need explaining here.

**Logits.** The table doesn't store probabilities directly. It stores raw scores called *logits*, which can be any number. A function called **softmax** turns a row of logits into probabilities. It makes them all positive and makes them add up to 1. Higher logit means higher probability. We store logits because they can be nudged freely in any direction without breaking the "adds up to 1" rule.

**Loss.** To learn, the model needs a score for how wrong it was. We use **cross-entropy loss**, which in plain English is *"how surprised were you by the real next character?"*

- Gave the right character 100% probability → loss 0 (no surprise)
- Gave it 50% → loss 0.69
- Gave it 1% → loss 4.6 (very surprised)

Mathematically it's `-log(probability you gave the right answer)`, averaged over every prediction.

## Training

The training loop in [`zoey/train.py`](../zoey/train.py) is shared by every chapter. Each step:

1. **Grab a batch.** 64 random windows of 256 characters each from TinyStories.
2. **Predict.** For every character in every window, look up the next-character probabilities.
3. **Measure.** Compute the loss: how surprised were we, on average?
4. **Nudge.** `loss.backward()` works out, for each of the 9,409 numbers, which direction would reduce the loss. The optimizer (AdamW) moves each one a small step that way.

Repeat 3,000 times. On a GPU this takes about 20 seconds.

Notice that one batch is 64 × 256 = **16,384 predictions**. Every position in a window is its own training example, because the model predicts the next character at every position at once.

Run it yourself:

```bash
uv run python -m zoey.train configs/01-bigram.toml
```

## Look inside: what to watch in the viewer

Open Chapter 1 in the viewer and press play.

**What TinyWriter writes.** At step 0 it's pure keyboard mash: `as1m{:W?QrdSQm`. Within 50 steps, spaces appear in sensible places and vowels and consonants start alternating. By the end, you get text like:

```
Once upon a time sas. ay, Tid w s the upiskedelansear. m he welicha " hales,
```

It's not English, but it *sounds* like English. There are real short words (*the*, *he*, *to*), capitals after periods, and plausible letter combinations. Every random choice is the same at every snapshot, so the changes you see come from the model learning, not from luck.

**Guess the next letter.** Click the `i` in *time*. TinyWriter has read `Once upon a t`, but a bigram model can only see the `t`. Its top guess is `h` (30%), because after `t` the most common letter in English is `h`. The real answer `i` gets only about 4%. **This is the bigram model's fundamental limitation:** it has no idea it's in the word *time*.

**The bigram table.** Select row `q` and press play. At step 0, the row is random noise and `u` has under 1%. By step 100 the `u` column has taken over (above 80%), and by the end it's essentially 100%. Try other rows: `.` learns that a space comes next, and vowels learn that consonants follow.

**Loss.** Three things to notice:

1. **TinyWriter starts worse than random guessing.** Uniform guessing over 97 characters gives a loss of ln(97) ≈ 4.57. TinyWriter starts at about 5.2, because its random initial table is *confidently wrong*.
2. **Almost all the learning happens in the first 100 steps.** The timeline uses a log scale so you can see this happen.
3. **The curve goes flat at about 2.29.** That's not a training problem. It's the best a bigram model can *possibly* do on this data. Seeing one letter back only gets you so far. To beat 2.29, the model needs to see more context, and that's where the rest of this course is headed.

## Exercises

1. **Learning rate.** Set `learning_rate = 0.005` in the config and retrain. How does the loss curve change? Then try `1.0`.
2. **Count instead.** Write a few lines that fill the table by counting character pairs in the training data, take the log, and load it into `model.table.weight`. Do you get the same loss of about 2.29? (You should get very close. Gradient descent and counting arrive at the same answer.)
3. **Temperature.** In [`zoey/sample.py`](../zoey/sample.py), generate text with `temperature=0.5` and `temperature=1.5`. What changes?

## What's next

The bigram model's biggest weakness is that it sees one character at a time, and characters themselves are a clumsy unit. In **Chapter 2** we build a real tokenizer (byte-pair encoding) that learns to treat common chunks like `the`, `ing`, and ` once` as single tokens.
