# Find best face

Use facial landmarks to get the best embedding possible.


# TODO

* Phase 0:
    - learn linear regression of facial landmarks to quality
    - Decide on simple, phase0 result metric
    - Visualise result
    - Measure speed of 
        - face detection
        - face embedding generation
        - best face 
        - Embedding generation and quality generation
            - => Maybe this can be done directly, without needing to learn anything

* Ideas to experiment on
    - Visualize one person faces, ordered by quality
    -  


* Phase 1-n:
    - save intermediate qualities for faster cycling
    - Use 68-point landmarks (in stead of 5)
    - Deeper quality metrics. Example
        - Compare only with best 20% of ohher faces
        - should use (mean - stddev) when comparing own and others.
        - 

* Make unit tests
