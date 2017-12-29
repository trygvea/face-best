# Find best face

From a set of face images for a certain person, find the face image that
  - Will match most other faces of the same person
  - Will match least faces of other persons

Strategies:
  - Use facial landmarks to learn the best embedding possible
  - Or, maybe just use embeddings and calculate a good quality metric (see below)


# WIP - Work in progress

## Premature conclusion

- After visualising faces ordered by quality, I cannot see what we can learn from landmarks - especially not landmark-5.
    A tiny doubt still exist for landmark-68, so maybe we should give it a try for the experience...

- A much better result will be obtained by calculating embeddings for all images.
  This will probably just be a factor of 2 or so slower. On an MBP 2016:
    - Detecting face bounds takes ~10 ms (needed for both landmark and embedding calculation)
    - Finding landmarks takes 1.5/3 ms (needed for embedding  calculation)
    - Calculating embeddings takes ~20 ms more.
    - All other calculation are relatively small in comparison
    -
  Calculation of embedding to quality for one face image will require:
    - Detect face bounds, finding landmarks, calculate embeddings: ~32 ms
    - Calculate quality of matches with not too many images of the same and other persons will
    - If wee keep a properly sized pool (100-1000?) of 'other' faces with embeddings nearby for quality measurments, the
      quality calcualtion

- Should probably work more on
    - quality measurement, involving std deviation, use just the best/worst subsets from own/others etc etc
    - Find a result metric that matches new strategy (use full embeddings for quality)
        TODO
        - Dont necessarily use the worst match of other, it is sufficcient that they are above a certain limit!!!!
    - (ms-celeb data quality) Filter images of wrongly placed persons
    - (use full ms-celeb dataset)

# TODO

* Align numbers on figures better
* Speed up figures  

* Phase 0:
    - learn linear regression of facial landmarks to quality?
        - Check out https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/
    - Decide on simple, phase0 result metric
    - Visualize result

* Phase 1-n:
    - get hold of full MS-celeb - (if we are learning something)  
    - save intermediate qualities for faster cycling
    - Use 68-point landmarks (in stead of 5)
    - Deeper quality metrics. Example
        - Compare only with best 20% of ohher faces
        - should use (mean - stddev) when comparing own and others.
    - Try to keep more than one embedding for better results!
        - Keep one for each age!

* Unit tests
