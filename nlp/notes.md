Caching model
---------------
* discounts n-gram weight (1-lambda) based on how "old" it is.
- works poorly for speech recognition

Stupid back off
----------------
* low frequency n-grams are boosted by dividing by 0.4x their n-1-gram count 
+ works as well as any other technique for large datasets
- not good for small datasets

Interpolation
-------------

