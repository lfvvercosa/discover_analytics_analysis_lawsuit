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
        * Run 
    - Step 3.3: Run <code> python3 2_main_finish_preprocessing 4 kmeans False </code> to generate the feature refering to K-means clustering
    - Step 3.4: Run <code> python3 2_main_finish_preprocessing 4 kmeans True </code> to merge the cluster's files in 'dataset/tribunais_trabalho/cluster_feat_all.csv'







- Esse projeto contém todo o código necessário para executar os experimentos do artigo entitulado "O Uso da Mineração de Processos na Análise do Tempo das Movimentações Processuais: Métricas e Desafios à Celeridade Processual no Judiciário Brasileiro".

- Passo 1: Instalar todas as bibliotecas respeitando o versionamento descrito no documento requirements  
  <code> pip install -r requirements.txt </code>
- Passo 2: Executar os quatro arquivos 'main' na ordem de sua numeração, isto é:  
  1. 1_main_create_log.py
  2. 2_main_create_dataset.py
  3. 3_main_apply_models.py
  4. 4_main_feat_import.py


