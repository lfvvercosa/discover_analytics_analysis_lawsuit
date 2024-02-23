def sel_subset_feat(gt):
    if gt == 'FITNESS':
        return ['FITNESS_50_RAND']
    
    if gt == 'PRECISION':
        return ['PRECISION_50_RAND']


def sel_top_feat(k, gt):
    if gt == 'PRECISION':
        return ['ABS_EDGES_ONLY_IN_MODEL_W']
    
    if gt == 'FITNESS':
        if k > 1:
            return ['ALIGNMENTS_MARKOV']
        else:
            return ['ALIGNMENTS_MARKOV_2_K1']


def sel_top_abs_feat(markov_k, gt):
    if markov_k == -1:
        if gt == 'FITNESS':
            return ['FOOTPRINT_COST_FIT_W']
        
        if gt == 'PRECISION':
            return ['FOOTPRINT_COST_PRE']
    else:
        if gt == 'FITNESS':
            return ['ABS_EDGES_ONLY_IN_LOG_W']
        
        if gt == 'PRECISION':
            return ['ABS_EDGES_ONLY_IN_MODEL_W']


def sel_top_10_abs_feat(markov_k, gt):
    if markov_k == 1:
        if gt == 'FITNESS':
            return [
                'ABS_EDGES_ONLY_IN_LOG_BTW',
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'ABS_MAX_DEGREE_DIV',
                'DIST_EDGES_PERCENT',
                'ABS_MODEL_DEGREE_ENTROPY',
                'ABS_EDGES_ONLY_IN_MODEL',
                'ABS_LINK_DENSITY_DIV',
                'ABS_MODEL_LINK_DENSITY',
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ABS_EDGES_ONLY_IN_LOG',
            ]
        
        if gt == 'PRECISION':
            return [
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ABS_MODEL_DEGREE_ENTROPY',
                'ABS_EDGES_ONLY_IN_LOG',
                'ABS_MODEL_LINK_DENSITY',
                'ABS_LOG_DEGREE_ENTROPY',
                'ABS_EDGES_ONLY_IN_LOG_CC',
                'ABS_EDGES_ONLY_IN_MODEL',
                'ABS_LINK_DENSITY_DIV',
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'ABS_EDGES_ONLY_IN_MODEL_W_2',
            ]

    if markov_k == 2:
        if gt == 'FITNESS':
            return [
                'ABS_EDGES_ONLY_IN_LOG_CC',
                'ABS_EDGES_ONLY_IN_LOG_BTW_NORM',
                'ABS_MAX_DEGREE_DIV',
                'ABS_DEGREE_ENTROPY_DIV',
                'ABS_EDGES_ONLY_IN_MODEL_W_2',
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'ABS_EDGES_ONLY_IN_MODEL',
                'ABS_MODEL_LINK_DENSITY',
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ABS_EDGES_ONLY_IN_LOG',
            ]
        
        if gt == 'PRECISION':
            return [
                'ABS_MAX_CLUSTER',
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ABS_EDGES_ONLY_IN_LOG',
                'ABS_MODEL_DEGREE_ENTROPY',
                'DIST_NODES_PERCENT',
                'DIST_EDGES_PERCENT',
                'ABS_MAX_CLUSTER_DIV',
                'ABS_EDGES_ONLY_IN_MODEL',
                'ABS_EDGES_ONLY_IN_MODEL_W_2',
                'ABS_EDGES_ONLY_IN_MODEL_W',
            ]
    
    if markov_k == 3:
        if gt == 'FITNESS':
            return [
                'ABS_EDGES_ONLY_IN_LOG_BTW',
                'ABS_EDGES_ONLY_IN_MODEL_W_2',
                'ABS_MODEL_DEGREE_ENTROPY',
                'ABS_DEGREE_ENTROPY_DIV',
                'ABS_DEGREE_ENTROPY',
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'DIST_EDGES_PERCENT',
                'ABS_EDGES_ONLY_IN_MODEL',
                'ABS_MAX_CLUSTER',
                'ABS_EDGES_ONLY_IN_LOG',
                'ABS_EDGES_ONLY_IN_LOG_W',
            ]
        
        if gt == 'PRECISION':
            return [
                'ABS_EDGES_ONLY_IN_LOG_BTW_NORM',
                'ABS_DEGREE_ENTROPY',
                'ABS_MAX_DEGREE',
                'ABS_EDGES_ONLY_IN_LOG_BTW',
                'DIST_NODES_PERCENT',
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ABS_EDGES_ONLY_IN_LOG',
                'DIST_EDGES_PERCENT',
                'ABS_EDGES_ONLY_IN_MODEL',
                'ABS_EDGES_ONLY_IN_MODEL_W_2',
                'ABS_EDGES_ONLY_IN_MODEL_W',
            ]


def sel_top_10_feat(markov_k, gt):
    if markov_k == 1:
        if gt == 'FITNESS':
            return [
                'ALIGNMENTS_MARKOV',
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ABS_EDGES_ONLY_IN_LOG',
                'LOG_PERC_UNIQUE_SEQS',
                'ABS_MAX_DEGREE',
                'ABS_MAX_DEGREE_DIV',
                'LOG_EVENTS_TYPES',
                'PN_PLACES',
                'ABS_LOG_MAX_DEGREE',
                'PN_TRANS',
  
            ]
        if gt == 'PRECISION':
            return [
                'ABS_EDGES_ONLY_IN_MODEL',
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'ABS_MODEL_MAX_DEGREE',
                'PN_ARCS_MEAN',
                'ABS_LINK_DENSITY_DIV',
                'ABS_MAX_DEGREE_DIV',
                'ABS_MODEL_LINK_DENSITY',
                'PN_TRANS',
                'PN_PLACES',
                'PN_SILENT_TRANS',
            ]
        if gt == 'TIME_MARKOV':
            return [
            'PN_ARCS_MAX',
                'ABS_DEGREE_ENTROPY',
                'ABS_DEGREE_ENTROPY_DIV',
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ABS_MODEL_DEGREE_ENTROPY',
                'PN_OUT_ARCS_MAX',
                'ABS_MODEL_MAX_DEGREE',
                'PN_SILENT_TRANS',
            ] 
    if markov_k == 2:
        if gt == 'FITNESS':
            return [
                'ALIGNMENTS_MARKOV',      
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ABS_EDGES_ONLY_IN_LOG',  
                'LOG_PERC_UNIQUE_SEQS',   
                'ABS_DEGREE_ENTROPY',     
                'ABS_DEGREE_ENTROPY_DIV', 
                'LOG_EVENTS_TYPES',       
                'ABS_MAX_DEGREE',         
                'ABS_MAX_DEGREE_DIV',     
                'PN_PLACES',
            ]
        if gt == 'PRECISION':
            return [
                'ABS_EDGES_ONLY_IN_MODEL',
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'ABS_MODEL_MAX_DEGREE',
                'PN_ARCS_MEAN',
                'ABS_MAX_DEGREE_DIV',
                'ABS_MODEL_DEGREE_ENTROPY',
                'ABS_MAX_DEGREE',
                'ABS_DEGREE_ENTROPY_DIV',
                'ABS_DEGREE_ENTROPY',
                'PN_SILENT_TRANS',
            ]
        if gt == 'TIME_MARKOV':
            return [
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'ABS_MODEL_MAX_DEGREE',
                'LOG_AVG_SEQS_LENGTH',
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ABS_LOG_DEGREE_ENTROPY',
                'PN_OUT_ARCS_MAX',
                'ABS_LOG_LINK_DENSITY',
                'ABS_MODEL_DEGREE_ENTROPY',
                'PN_SILENT_TRANS',
            ]
    if markov_k == 3:
        if gt == 'FITNESS':
            return [
                'ALIGNMENTS_MARKOV',
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ABS_EDGES_ONLY_IN_LOG',
                'LOG_PERC_UNIQUE_SEQS',
                'LOG_EVENTS_TYPES',
                'PN_PLACES',
                'ABS_MAX_DEGREE',
                'PN_TRANS',
                'ABS_LOG_LINK_DENSITY',
                'ABS_LINK_DENSITY_DIV',
            ]
        if gt == 'PRECISION':
            return [
                'ABS_EDGES_ONLY_IN_MODEL',           
                'ABS_EDGES_ONLY_IN_MODEL_W',         
                'ABS_MODEL_MAX_DEGREE',              
                'ABS_MAX_DEGREE_DIV',                
                'PN_ARCS_MEAN',                      
                'ABS_DEGREE_ENTROPY',                
                'ABS_DEGREE_ENTROPY_DIV',            
                'ABS_MODEL_DEGREE_ENTROPY',          
                'ABS_MAX_DEGREE',                    
                'ABS_MODEL_LINK_DENSITY',

            ]
        if gt == 'TIME_MARKOV':
            return [
                'ABS_MAX_DEGREE_DIV',
                'ABS_MODEL_LINK_DENSITY',
                'ABS_DEGREE_ENTROPY_DIV',
                'ABS_DEGREE_ENTROPY',
                'ABS_MODEL_MAX_DEGREE',
                'ABS_LINK_DENSITY_DIV',
                'ABS_EDGES_ONLY_IN_LOG_BTW_NORM',
                'PN_SILENT_TRANS',
                'ABS_MODEL_DEGREE_ENTROPY',
            ]

def select_rf_features(markov_k, gt):
    if markov_k == 1:
        if gt == 'FITNESS':
            return [
                'ALIGNMENTS_MARKOV', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'PN_SILENT_TRANS',
                'PN_TRANS',
                'LOG_SEQS',
                'LOG_EVENTS',
                'PN_ARCS_MEAN',
                'PN_PLACES',
                'LOG_AVG_EDIT_DIST',
                'ABS_LINK_DENSITY_DIV',
            ]
        if gt == 'PRECISION':
            return [
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'ABS_EDGES_ONLY_IN_MODEL',
                'PN_ARCS_MEAN',
                'ABS_LINK_DENSITY_DIV',
                'LOG_AVG_SEQS_LENGTH',
                'LOG_UNIQUE_SEQS',
                'PN_SILENT_TRANS',
                'LOG_AVG_EDIT_DIST',
                'LOG_SEQS',
                'ABS_MODEL_LINK_DENSITY',
                'ABS_EDGES_ONLY_IN_LOG_CC',
            ] 
    if markov_k == 2:
        if gt == 'FITNESS':
            return [
                'ALIGNMENTS_MARKOV', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'PN_ARCS_MEAN',
                'LOG_EVENTS',
                'LOG_SEQS',
                'PN_SILENT_TRANS',
                'PN_TRANS',
                'PN_PLACES',
                'LOG_UNIQUE_SEQS',
                'ABS_EDGES_ONLY_IN_LOG_CC',
            ]
        if gt == 'PRECISION':
            return [
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'ABS_EDGES_ONLY_IN_MODEL',
                'LOG_AVG_SEQS_LENGTH',
                'PN_ARCS_MEAN',
                'LOG_AVG_EDIT_DIST',
                'LOG_SEQS',
                'PN_SILENT_TRANS',
                'LOG_UNIQUE_SEQS',
                'ABS_MODEL_DEGREE_ENTROPY',
                'LOG_EVENTS',
                'ABS_EDGES_ONLY_IN_LOG_W',
            ]
    if markov_k == 3:
        if gt == 'FITNESS':
            return [
                'ALIGNMENTS_MARKOV', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'LOG_EVENTS',
                'PN_ARCS_MEAN',
                'LOG_SEQS',
                'PN_TRANS',
                'LOG_UNIQUE_SEQS',
                'PN_SILENT_TRANS',
                'PN_PLACES',
                'ABS_DEGREE_ENTROPY_DIV',
            ]
        if gt == 'PRECISION':
            return [
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'ABS_EDGES_ONLY_IN_MODEL',
                'LOG_AVG_SEQS_LENGTH',
                'LOG_SEQS',
                'PN_ARCS_MEAN',
                'LOG_AVG_EDIT_DIST',
                'PN_SILENT_TRANS',
                'LOG_UNIQUE_SEQS',
                'ABS_MAX_DEGREE_DIV',
                'ABS_MAX_DEGREE',
                'ABS_MODEL_DEGREE_ENTROPY',
            ]

def select_xgb_features(markov_k, gt):
    if markov_k == 1:
        if gt == 'FITNESS':
            return [
                'ABS_EDGES_ONLY_IN_LOG', 
                'PN_SILENT_TRANS',
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ALIGNMENTS_MARKOV', 
                'PN_TRANS',
                'LOG_EVENTS_TYPES',
                'LOG_SEQS',
                'PN_PLACES',
                'LOG_EVENTS',
                'ABS_MAX_DEGREE_DIV',
                'ABS_EDGES_ONLY_IN_MODEL',
            ]
        if gt == 'PRECISION':
            return [
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'LOG_AVG_SEQS_LENGTH',
                'LOG_UNIQUE_SEQS',
                'ABS_LINK_DENSITY_DIV',
                'LOG_SEQS',
                'PN_ARCS_MEAN',
                'ABS_LOG_LINK_DENSITY',
                'ABS_EDGES_ONLY_IN_MODEL',
                'ABS_MODEL_MAX_DEGREE',
                'ABS_LOG_MAX_DEGREE',
                'ABS_MAX_DEGREE',
            ] 
    if markov_k == 2:
        if gt == 'FITNESS':
            return [
                'ALIGNMENTS_MARKOV', 
                'PN_TRANS',
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_DEGREE_ENTROPY',
                'ABS_EDGES_ONLY_IN_MODEL_W', 
                'ABS_EDGES_ONLY_IN_LOG_CC', 
                'ABS_EDGES_ONLY_IN_MODEL',
                'PN_PLACES',
                'PN_ARCS_MEAN',
                'LOG_EVENTS',
            ]
        if gt == 'PRECISION':
            return [
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'LOG_SEQS',
                'ABS_EDGES_ONLY_IN_MODEL',
                'ABS_LOG_DEGREE_ENTROPY',
                'ABS_MAX_DEGREE_DIV',
                'PN_SILENT_TRANS',
                'ABS_EDGES_ONLY_IN_LOG_W',
                'PN_ARCS_MEAN',
                'ABS_MODEL_LINK_DENSITY',
                'ABS_MAX_DEGREE',
                'LOG_PERC_UNIQUE_SEQS',

            ]
    if markov_k == 3:
        if gt == 'FITNESS':
            return [
                'ALIGNMENTS_MARKOV', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'PN_TRANS',
                'LOG_AVG_SEQS_LENGTH',
                'ABS_DEGREE_ENTROPY',
                'LOG_SEQS',
                'PN_PLACES',
                'LOG_EVENTS',
                'ABS_EDGES_ONLY_IN_MODEL',
                'ABS_EDGES_ONLY_IN_MODEL_W', 
            ]
        if gt == 'PRECISION':
            return [
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'LOG_SEQS',
                'ABS_EDGES_ONLY_IN_MODEL',
                'LOG_AVG_SEQS_LENGTH',
                'LOG_PERC_UNIQUE_SEQS',
                'ABS_EDGES_ONLY_IN_LOG_W',
                'PN_SILENT_TRANS',
                'PN_ARCS_MEAN',
                'ABS_LINK_DENSITY_DIV',
                'ABS_DEGREE_ENTROPY',
                'ABS_MAX_CLUSTER',
            ]


def sel_top_7_rf_feat(markov_k, gt):
    if markov_k == 1:
        if gt == 'FITNESS':
            return [
                'PN_TRANS', 
                'PN_PERC_SILENT_TRANS', 
                'PN_SILENT_TRANS', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ALIGNMENTS_MARKOV',

            ]
        if gt == 'PRECISION':
            return[
                'variant_entropy_norm', 
                'LOG_AVG_SEQS_LENGTH', 
                'sequence_entropy_norm', 
                'ABS_LINK_DENSITY_DIV', 
                'ABS_EDGES_ONLY_IN_MODEL', 
                'ABS_EDGES_ONLY_IN_MODEL_W',
                'ABS_EDGES_ONLY_IN_MODEL_W_2', 

            ]

    if markov_k == 2:
        if gt == 'FITNESS':
            return[
                'LOG_SEQS', 
                'LOG_EVENTS', 
                'PN_PERC_SILENT_TRANS', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ABS_EDGES_ONLY_IN_LOG',
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'PN_SILENT_TRANS', 
                'variant_entropy_norm', 
                'LOG_AVG_SEQS_LENGTH',
                'sequence_entropy_norm', 
                'ABS_EDGES_ONLY_IN_MODEL', 
                'ABS_EDGES_ONLY_IN_MODEL_W_2', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]

    if markov_k == 3:
        if gt == 'FITNESS':
            return [
                'PN_PERC_SILENT_TRANS', 
                'LOG_SEQS', 
                'LOG_EVENTS', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'LOG_AVG_SEQS_LENGTH', 
                'variant_entropy_norm', 
                'sequence_entropy_norm', 
                'DIST_EDGES_PERCENT', 
                'ABS_EDGES_ONLY_IN_MODEL', 
                'ABS_EDGES_ONLY_IN_MODEL_W_2', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]



def sel_top_5_rf_feat(markov_k, gt):
    if markov_k == 1:
        if gt == 'FITNESS':
            return[
                'PN_SILENT_TRANS', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'ABS_EDGES_ONLY_IN_MODEL', 
                'sequence_entropy_norm', 
                'ABS_LINK_DENSITY_DIV', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 
                'ABS_EDGES_ONLY_IN_MODEL_W_2', 

            ]
    
    if markov_k == 2:
        if gt == 'FITNESS':
            return[
                'PN_PERC_SILENT_TRANS', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'LOG_AVG_SEQS_LENGTH', 
                'sequence_entropy_norm', 
                'ABS_EDGES_ONLY_IN_MODEL', 
                'ABS_EDGES_ONLY_IN_MODEL_W_2', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]
    if markov_k == 3:
        if gt == 'FITNESS':
            return[
                'LOG_EVENTS', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'DIST_EDGES_PERCENT', 
                'sequence_entropy_norm', 
                'ABS_EDGES_ONLY_IN_MODEL', 
                'ABS_EDGES_ONLY_IN_MODEL_W_2',
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]


def sel_top_3_rf_feat(markov_k, gt):
    if markov_k == 1:
        if gt == 'FITNESS':
            return[
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'ABS_LINK_DENSITY_DIV', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 
                'ABS_EDGES_ONLY_IN_MODEL_W_2', 

            ]   
    if markov_k == 2:
        if gt == 'FITNESS':
            return[
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ALIGNMENTS_MARKOV', 

            ] 
        if gt == 'PRECISION':
            return[
                'ABS_EDGES_ONLY_IN_MODEL', 
                'ABS_EDGES_ONLY_IN_MODEL_W_2', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]   
    if markov_k == 3:
        if gt == 'FITNESS':
            return[
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ALIGNMENTS_MARKOV', 

            ] 
        if gt == 'PRECISION':
            return[
                'ABS_EDGES_ONLY_IN_MODEL', 
                'ABS_EDGES_ONLY_IN_MODEL_W_2',
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]


def sel_treshold15_rf_feat(markov_k, gt):
    if markov_k == 1:
        if gt == 'FITNESS':
            return[
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'FOOTPRINT_COST_FIT_W',
                'ABS_EDGES_ONLY_IN_LOG', 
                'ALIGNMENTS_MARKOV',

            ]
        if gt == 'PRECISION':
            return[
                'sequence_entropy_norm', 
                'ABS_EDGES_ONLY_IN_MODEL', 
                'ABS_LINK_DENSITY_DIV', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 
                'ABS_EDGES_ONLY_IN_MODEL_W_2',

            ]
    if markov_k == 2:
        if gt == 'FITNESS':
            return[
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'ABS_EDGES_ONLY_IN_MODEL', 
                'ABS_EDGES_ONLY_IN_MODEL_W_2',
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]
    if markov_k == 3:
        if gt == 'FITNESS':
            return[
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]

def sel_top_7_xgb_feat(markov_k, gt): 
    if markov_k == 1:
        if gt == 'FITNESS':
            return[
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'PN_PERC_SILENT_TRANS',
                'LOG_AVG_EDIT_DIST', 
                # 'ALIGNMENTS_MARKOV',
                'ALIGNMENTS_MARKOV_2_K1',
                'PN_SILENT_TRANS', 
                'FOOTPRINT_COST_FIT_W',
                'ABS_EDGES_ONLY_IN_LOG', 

            ]
        if gt == 'PRECISION':
            return[
                'ABS_EDGES_ONLY_IN_MODEL', 
                'PN_TRANS', 
                'PN_ARCS_MEAN',
                'variant_entropy_norm', 
                'PN_IN_ARCS_INV_TRAN_MEAN', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 
                'sequence_entropy_norm', 

            ]   

    if markov_k == 2:
        if gt == 'FITNESS': 
            return[
                'ABS_DEGREE_ENTROPY', 
                'PN_TRANS', 
                'PN_PERC_SILENT_TRANS', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_EDGES_ONLY_IN_LOG_W',
                'ALIGNMENTS_MARKOV',
                # 'ALIGNMENTS_MARKOV_2_K1', 

            ]
        if gt == 'PRECISION':
            return[
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ABS_LOG_DEGREE_ENTROPY', 
                'LOG_SEQS',
                'LOG_AVG_SEQS_LENGTH', 
                'ABS_EDGES_ONLY_IN_MODEL', 
                'sequence_entropy_norm', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]
    if markov_k == 3:
        if gt == 'FITNESS':
            return[
                'ABS_DEGREE_ENTROPY', 
                'FOOTPRINT_COST_FIT_W', 
                'LOG_AVG_SEQS_LENGTH', 
                'PN_PERC_SILENT_TRANS', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ABS_EDGES_ONLY_IN_LOG',
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'LOG_SEQS', 
                'variant_entropy_norm', 
                'LOG_AVG_SEQS_LENGTH', 
                'ABS_EDGES_ONLY_IN_MODEL',
                'sequence_entropy_norm', 
                'LOG_PERC_UNIQUE_SEQS',
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]   

def sel_top_5_xgb_feat(markov_k, gt): 
    if markov_k == 1:
        if gt == 'FITNESS':
            return[
                'LOG_AVG_EDIT_DIST', 
                'ALIGNMENTS_MARKOV', 
                'PN_SILENT_TRANS', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 

            ]
        if gt == 'PRECISION':
            return[
                'PN_ARCS_MEAN', 
                'variant_entropy_norm', 
                'PN_IN_ARCS_INV_TRAN_MEAN', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 
                'sequence_entropy_norm', 

            ]   
    if markov_k == 2:
        if gt == 'FITNESS':
            return[
                'PN_PERC_SILENT_TRANS', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ALIGNMENTS_MARKOV', 

            ]   
        if gt == 'PRECISION':
            return[
                'LOG_SEQS', 
                'LOG_AVG_SEQS_LENGTH', 
                'ABS_EDGES_ONLY_IN_MODEL', 
                'sequence_entropy_norm', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]   
    if markov_k == 3:
        if gt == 'FITNESS':
            return[
                'LOG_AVG_SEQS_LENGTH', 
                'PN_PERC_SILENT_TRANS', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ALIGNMENTS_MARKOV', 

            ]   
        if gt == 'PRECISION':
            return[
                'LOG_AVG_SEQS_LENGTH', 
                'ABS_EDGES_ONLY_IN_MODEL', 
                'sequence_entropy_norm', 
                'LOG_PERC_UNIQUE_SEQS', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]   

def sel_top_3_xgb_feat(markov_k, gt):  
    if markov_k == 1:
        if gt == 'FITNESS':
            return[
                'PN_SILENT_TRANS', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 

            ]   
        if gt == 'PRECISION':
            return[
                'PN_IN_ARCS_INV_TRAN_MEAN', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 
                'sequence_entropy_norm', 

            ]   
    if markov_k == 2:
        if gt == 'FITNESS':
            return[
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'ABS_EDGES_ONLY_IN_MODEL', 
                'sequence_entropy_norm', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]
    if markov_k == 3:
        if gt == 'FITNESS':
            return[
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'sequence_entropy_norm', 
                'LOG_PERC_UNIQUE_SEQS', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]

def sel_treshold15_xgb_feat(markov_k, gt):
    if markov_k == 1:
        if gt == 'FITNESS':
            return[
                'LOG_AVG_EDIT_DIST', 
                'ALIGNMENTS_MARKOV', 
                'PN_SILENT_TRANS', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 

            ]
        if gt == 'PRECISION':
            return[
                'ABS_EDGES_ONLY_IN_MODEL', 
                'PN_TRANS', 
                'PN_ARCS_MEAN', 
                'variant_entropy_norm', 
                'PN_IN_ARCS_INV_TRAN_MEAN', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 
                'sequence_entropy_norm', 

            ]
    if markov_k == 2:
        if gt == 'FITNESS':
            return[
                'PN_TRANS', 
                'PN_PERC_SILENT_TRANS', 
                'FOOTPRINT_COST_FIT_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION':
            return[
                'ABS_EDGES_ONLY_IN_MODEL', 
                'sequence_entropy_norm', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]
    if markov_k == 3:
        if gt == 'FITNESS':
            return[
                'PN_PERC_SILENT_TRANS', 
                'ABS_EDGES_ONLY_IN_LOG_W', 
                'ABS_EDGES_ONLY_IN_LOG', 
                'ALIGNMENTS_MARKOV', 

            ]
        if gt == 'PRECISION': 
            return[
                'ABS_EDGES_ONLY_IN_MODEL', 
                'sequence_entropy_norm', 
                'LOG_PERC_UNIQUE_SEQS', 
                'ABS_EDGES_ONLY_IN_MODEL_W', 

            ]                               




def translate_features_name(feat):
    mapping = {
        'ALIGNMENTS_MARKOV':'(G4)f1',
        'ABS_EDGES_ONLY_IN_LOG':'(G1)f1',
        'ABS_EDGES_ONLY_IN_LOG_W':'(G1)f2',
        'ABS_EDGES_ONLY_IN_LOG_BTW':'(G1)f3',
        'ABS_EDGES_ONLY_IN_LOG_BTW_NORM':'(G1)f4',
        'ABS_EDGES_ONLY_IN_LOG_CC':'(G1)f5',
        'ABS_EDGES_ONLY_IN_MODEL':'(G1)f6',
        'ABS_EDGES_ONLY_IN_MODEL_W':'(G1)f7',
        'ABS_MAX_DEGREE':'(G1)f8',
        'ABS_MAX_DEGREE_DIV':'(G1)f9',
        'ABS_MAX_CLUSTER':'(G1)f10',
        'ABS_MAX_CLUSTER_DIV':'(G1)f11',
        'ABS_DEGREE_ENTROPY':'(G1)f12',
        'ABS_DEGREE_ENTROPY_DIV':'(G1)f13',
        'ABS_LINK_DENSITY_DIV':'(G1)f14',
        'ABS_LOG_MAX_DEGREE':'(G1)f15',
        'ABS_LOG_DEGREE_ENTROPY':'(G1)f16',
        'ABS_LOG_LINK_DENSITY':'(G1)f17',
        'ABS_MODEL_MAX_DEGREE':'(G1)f18',
        'ABS_MODEL_DEGREE_ENTROPY':'(G1)f19',
        'ABS_MODEL_LINK_DENSITY':'(G1)f20',
        'PN_SILENT_TRANS':'(G3)f1',
        'PN_TRANS':'(G3)f2',
        'PN_PLACES':'(G3)f3',
        'PN_ARCS_MEAN':'(G3)f4',
        'PN_ARCS_MAX':'(G3)f5',
        'LOG_EVENTS':'(G2)f1',
        'LOG_EVENTS_TYPES':'(G2)f2',
        'LOG_SEQS':'(G2)f3',
        'LOG_UNIQUE_SEQS':'(G2)f4',
        'LOG_PERC_UNIQUE_SEQS':'(G2)f5',
        'LOG_AVG_SEQS_LENGTH':'(G2)f6',
        'LOG_AVG_EDIT_DIST':'(G2)f7',
    }

    new_names = []

    for f in feat:
        new_names.append(mapping[f])
    
    return new_names