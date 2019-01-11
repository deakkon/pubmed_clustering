# pubmed_clustering
Solution to coding part of the interview process for Invitae. Position software engineer, Scientific IR. 

The solution was developed with Python 3.6 and the Anaconda Python Distribution on Windows OS. 

Installation steps:

1. OPTIONAL create and activate a new virtual environment:

    With Anaconda:
    
        conda create -n env_name python=3.6
        activate env_name
        
    Alternatively:
    
        python3 -m venv env_name -env
        source env_name/bin/activate
        
2. pip install -r requirements.txt

    As some of the dependencies are not available in the conda repo, we use pip to install all libraries.

3. run the code:

    a. python main.py -m train
            
         Executves input feature space xploration, traines and serializes seperate HDBSCAN and KMeans for individual input components:['title'],
                                 ['abstract'],
                                 ['title', 'abstract'],
                                 ['title', 'NE'],
                                 ['title', 'abstract', 'NE']
                                 
         For each invidiual input component, ngrams in range frm (1,1) to (1,4) are evaluated. Best performing n-gram is serlized for each input component. 
                                 
         After training and serialization, results are available:
            I. as a csv file in the folder reports/results/results.csv, where for each input components Homogenity score, Adjustem Mutual Information and inter- and intra-cluster varianc information is stored. 
            II. Visualized with help of tSNE, available in img/
            III. Clustring results, obtained by applying the best perfroming combinations for both HDBSCAN and K-Means, available in report/cluster_reports/clusters_ground_truth_HDBSCAN.tsv and report/cluster_reports/clusters_ground_truth_KMEANS.tsv.
            IV. Serialized models, available in prepared/utils. Models are serialized with sklearn.external.joblib.
            
            Additionally, analysis reports for the two evaluated clustering algorithms are shown on the screen, including HS and AMI values, top k (k==10 by default) words per cluster and Cluster ID with PMIDs in individual cluster. 
            
    b. python main.py -m test  
    
        Executes inference on the test set (i.e. unlabeled ground truth) with the best perfomring HDBSCAN and K-Means algorthms. ['title', 'abstract', 'NE'] is used as the default input component.s. 
        
        After inference, the results (same format and content as above) are available in report/cluster_reports/clusters_test_set_HDBSCAN.tsv and report/cluster_reports/clusters_test_set_KMEANS.tsv.
        
        Additionally, analysis reports for the two evaluated clustering algorithms are shown on the screen, including HS and AMI values, top k (k==10 by default) words per cluster and Cluster ID with PMIDs in individual cluster. 
        
    c. python main.py -m external -i input_file -n corpus_name 
     
     with:  
       
        -i: path to input file, in the same format as unlabeled ground truth corpus
        
        -n: (optional) name of the corpus. If no name is given external_{timestamp} is used. 
        
        Generated reports are available in report/cluster_reports/cöuster_{corpus_name}_{timestamp}_HDBSCAN.csv and report/cluster_reports/cöuster_{corpus_name}_{timestamp}_KMEANS.csv. 
        
4. Read the report, to be found in report/pubmed_clustering.pdf
        
    
            
            
    
    
    
