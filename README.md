This is an experimental LLM.
Just a vehicle to experiment with.

TODOs:

more accuracy:
- multi-stage:
    - have a specialized token learner
        - if a different word is used in the same context,
          they are simular
                - in Paris the best Museum is the Louvre
                - in Brussel the best Museum is the XYZ
                - The Citymuseum is the best Museum in Amsterdam
                - inside Berlin the worst Museum is the ABCD
                - The Chatte is the worst Museum in Paris
              ==>  the vs The ==> same catagory
              ==>  best/worst ==> adjective catagory
              ==>  Paris/Brussel ==> city catagory

    - embedding support
        - learn catagories
        - use categories instead of token-IDs
          token-ids are only for the token learner
          the main predictor uses catagory IDss
- gaussian activation
- transformer arch
- check impact of more layers and/or wider layers
- multi epoch learning

more features:
- check prompt mode

be faster:
- optimistic concurrency
- CI for testing various options overnight
- OpenCL instead of OpenMP
