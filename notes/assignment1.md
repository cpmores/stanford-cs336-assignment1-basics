# Notes for Assigment 1

Unicode Standard:

+ a text encoding standard that maps characters to integer code points

we use utf-8 to divide unicode mapping to bytes

```python
>>> list('你'.encode('utf-8'))
[228, 189, 160]
>>> ord('你')
20320
```

Still not enough, byte-level tokenization boost the number of word-level tokenization, which may lead to more computation.
We need to map the most used words to something smaller and shorter.

So we train `BPE` to get a better mapping to compress the raw byte-level tokenization.

## BPE Tokenizer Training

Three steps:

+ **Vocabulary Initialization**: one-to-one mapping from bytestring token to integer ID. Our initial vocabulary if size 256 for byte-level training.
+ **Pre-tokenization**: What BPE can do is only merging words, human can divide the bytestring before it is sent to BPE training to ensure the continuous of some specific words. For example, build a pre-tokenization map above original bytestring: "t e x t t e s t", we got "test" for once, "text" for once, we only need to search words and multiple times it appears, we can get the appeared time of (t, e) byte-pair.
+ **Compute BPE merges**: after initialization and pre-tokenization, begin merging
