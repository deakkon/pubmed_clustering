# Pubmed Clustering Problem

These PMIDs are the partial results of PubMed searches for different diseases which were subsequently combined and shuffled. We would like you to retrieve the abstracts for those PMIDs and cluster them into groups that would ideally match the search query results.

Three files have been provided for you: 

1. pmids_gold_set_labeled.txt contains a 'gold set' of PMIDs labeled with the search terms used to retrieve them
2. pmids_gold_set_unlabeled.txt contains the same gold set of PMIDS, combined and shuffled, but with labels removed.
3. pmids_test_set_unlabeled.txt contains a separate 'test set' of PMIDs with no search term labels. Note that this includes a different set of diseases from gold set.

Please create a git repository hosted on Github or Bitbucket that contains your code for retrieving the abstracts and generating the clusters.  The repo should contain a README that provides instructions on how to build and run your code in the correct environment with dependencies installed (you might use e.g. a virtual environment or a docker container). Please assume we'll be running the code on a default Ubuntu 16.04 instance. The code will take as input the list of PMIDs and output the clusters in a file with the same format as pmids_gold_set_labeled.txt.

Describe:
1. The text processing steps used (tokenization, stemming, etc).
2. The similarity metric employed for the clustering algorithm.
3. The clustering algorithm chosen.
4. The parameters tested/used to execute the clustering algorithm.
5. Design choices and computational complexity of your code.
6. Quantitative metrics describing how your method performed on the gold set.
7. Run your approach on the test set as well and provide the output to us.
8. Provide a discussion of how your method performed. Where did it work well? Where did it not work well?
