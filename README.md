# Investigation of Lawsuit Process Duration: A Machine Learning and Process Mining Approach

This github project presents the code used for the paper submited to the journal Discover Analytics in the collection "Process Mining and Predictive Business Process Monitoring". In the following, we present the necessary steps to run this code.
 
- Step 1: Install all required libraries using <code> pip3 install -r requirements.txt </code>

- Step 2: Download the xes file named 'TRT_raw.xes' available in 'https://data.4tu.nl/datasets/fcdc27b9-44fd-476f-9a2d-1774e96e505f/3'. Place it inside dataset/tribunais_trabalho/ directory.
    - Alternatively, you can create the downloaded file by downloading the zip file named 'justica_trabalho' available in 'https://data.4tu.nl/datasets/fcdc27b9-44fd-476f-9a2d-1774e96e505f/2' and running the event log creation code  <code> python3 0_main_create_log.py  </code>, followed by <code> python3 1_main_finish_preprocessing  </code>. This may take a while.

- Step 3: The clustering features are already available at 'dataset/tribunais_trabalho/cluster_feat_all.csv'. However, it is also possible to generate them by running the following codes. If you do not wish to manually generate them, proceed to the next step.
    - Step 3.1: Run <code> python3 2_main_finish_preprocessing 4 agglom False </code> to generate the feature refering to agglomerative clustering
    - Step 3.2: Download actitrac jars from google drive 
        * <code> gdown --fuzzy 'https://drive.google.com/file/d/1YbcG-7lv2HGX9KUSfCj_ekEcUi1JuarD/view?usp=sharing' </code>
        * <code> gdown --fuzzy 'https://drive.google.com/file/d/1Z47RjZ8t9YhFihzwozZxqIWUtTbSlZjf/view?usp=sharing' </code>
        * Run <code> python3 2_main_finish_preprocessing 4 actitrac False </code> to generate the feature refering to ActiTraC clustering
    - Step 3.3: Run <code> python3 2_main_finish_preprocessing 4 kmeans False </code> to generate the feature refering to K-means clustering
    - Step 3.4: Run <code> python3 2_main_finish_preprocessing 4 kmeans True </code> to merge the cluster's files in 'dataset/tribunais_trabalho/cluster_feat_all.csv'

- Step 4: Run <code> python3 3_main_create_dataset.py </code> to create the raw dataset

- Step 5: Run <code> python3 4_main_process_dataset.py </code> to preprocess the raw dataset and create the model dataset

- Step 6: Run <code> python3 5_main_correlation.py </code> to exhibit the Pearson and Spearman correlations between the dataset features and the duration ground truth

- Step 7: In this step we run the machine learning algorithms to generate the time prediction results. Run:

    <code> python3 6_main_apply_models "['linear_reg','lgbm','svr']" </code> 
 
    to train and test the linear regression, DART and SVR models. Notice that we have made a modest parameter choice for increasing performance. However, they can be changed directly in the code to match the article configuration.

