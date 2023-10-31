# Python face recogniniton using PCA and LBPH
 A face recoginiton software that uses PCA and LBPH methods.

 Principal component analysis:[ https://en.wikipedia.org/wiki/Principal_component_analysis](https://en.wikipedia.org/wiki/Principal_component_analysis)

 Local binary patterns: [https://en.wikipedia.org/wiki/Local_binary_patterns](https://en.wikipedia.org/wiki/Local_binary_patterns)

 ## How to use:
 - Run `main/datasetcreator.py` to create a dataset for extracting PCA and LBPH data. Define the `user_id` and the desired number of samples for each data set creation sequence.
 - Run `training.py` to process the generated database. The data will be saved into the `recognizer` directory.
 - Run `detector.py` to use the software.
 
 The program currently supports up to 5 users. One can increase the number of users in the `detector.py` script.

 Note: Run wipe_out.py to delete the entire dataset and recognizer data.
