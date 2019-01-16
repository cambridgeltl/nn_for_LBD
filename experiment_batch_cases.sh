#!/bin/sh

#Determine which parts of the pipeline are executed
setup_experiment=False
create_representations=False
do_lbd=True

#Various settings
#options: 'closed_discovery_with_aggregators' 'closed_discovery_without_aggregators' 'open_discovery_with_aggregators_and_accumulators' 'open_discovery_without_aggregators_and_accumulators'
lbd_method='closed_discovery_without_aggregators'
embedding_creation_method='line' #Options: 'node2vec, 'line'

for dataset in lion_cs1 lion_cs2 lion_cs3 lion_cs4 lion_cs5 swan_mig_mag swan_som_arg swan_alz_est swan_alz_ind swan_sch_cip
do
    echo "\n----------------------------------------------------------"
    echo "----------------------------------------------------------"
    echo "----------------------------------------------------------"
    echo "Using dataset ${dataset}."

    ci='0:1:2:3'
    cl=':START_ID,:END_ID,year:int,metric_jaccard:float[]'

    if [ "$dataset" = "lion_cs1" ]
    then
        #LION case 1: NF-κB (PR:000001754) &	Bcl-2 (PR:000002307) &	Adenoma (MESH:D000236).
        #Van Der Heijden et al. (2016)
        datapath='data/PR000001754_MESHD000236'
        embeddingsshortname='cs1'
        a_node='PR:000001754'
        c_node='MESH:D000236'
        gold_b='PR:000002307'
        cutoff_year='2011'
    elif [ "$dataset" = "lion_cs2" ]
    then
        #LION case 2: NOTCH1 (PR:000011331) &	senescence (HOC:42) &	C/EBPβ (PR:000005308).
        #Hoare et al. (2016)
        datapath='data/PR000011331_PR000005308'
        embeddingsshortname='cs2'
        a_node='PR:000011331'
        c_node='PR:000005308'
        gold_b='HOC:42'
        cutoff_year='2011'
    elif [ "$dataset" = "lion_cs3" ]
    then
        #LION case 3: IL-17 (PR:000001138) &	p38α (PR:000003107) &	MKP-1 (PR:000006736).
        #Gaffen and McGeachy (2015)
        datapath='data/PR000001138_PR000006736'
        embeddingsshortname='cs3'
        a_node='PR:000001138'
        c_node='PR:000006736'
        gold_b='PR:000003107'
        cutoff_year='2010'
    elif [ "$dataset" = "lion_cs4" ]
    then
        #LION case 4: Nrf2 (PR:000011170) &	ROS (CHEBI:26523) &	pancreatic cancer (MESH:D010190).
        #DeNicola et al. (2011)
        datapath='data/PR000011170_MESHD010190'
        embeddingsshortname='cs4'
        a_node='PR:000011170'
        c_node='MESH:D010190'
        gold_b='CHEBI:26523'
        cutoff_year='2006'
    elif [ "$dataset" = "lion_cs5" ]
    then
        #LION case 5: CXCL12 (PR:000006066) & senescence (HOC:42) & thyroid cancer (MESH:D013964).
        #Kim et al. (2017)
        datapath='data/PR000006066_MESHD013964'
        embeddingsshortname='cs5'
        a_node='PR:000006066'
        c_node='MESH:D013964'
        gold_b='HOC:42'
        cutoff_year='2012'
    elif [ "$dataset" = "swan_mig_mag" ]
    then
        #Swanson case 1: Migraine & Magnesium (Swanson, 1988)
        datapath='data/MESHD008881_MESHD008274'
        embeddingsshortname='swan1'
        a_node='MESH:D008881'
        c_node='MESH:D008274'
        gold_b='b'
        cutoff_year='1983'
    elif [ "$dataset" = "swan_som_arg" ]
    then
        #Swanson case 2: Somatomedin & Arginine (Swanson, 1990)
        datapath='data/PR000009182_CHEBI29016'
        embeddingsshortname='swan2'
        a_node='PR:000009182'
        c_node='CHEBI:29016'
        gold_b='b'
        cutoff_year='1985'
    elif [ "$dataset" = "swan_alz_est" ]
    then
        #Swanson case 3: Alzheimer's & Estrogen (Smalheiser and Swanson, 1996)
        datapath='data/MESHD000544_MESHD004967'
        embeddingsshortname='swan3'
        a_node='MESH:D000544'
        c_node='MESH:D004967'
        gold_b='b'
        cutoff_year='1991'
    elif [ "$dataset" = "swan_alz_ind" ]
    then
        #Swanson case 4: Alzheimer's & Indomethacin (Smalheiser and Swanson, 1996)
        datapath='data/MESHD000544_MESHD007213'
        embeddingsshortname='swan4'
        a_node='MESH:D000544'
        c_node='MESH:D007213'
        gold_b='b'
        cutoff_year='1991'
    elif [ "$dataset" = "swan_sch_cip" ]
    then
        #Swanson case 5: Schizophrenia & Calcium Independent phospholipase A2
        #(Smalheiser and Swanson, 1998)
        datapath='data/MESHD012559_PR000012942'
        embeddingsshortname='swan5'
        bipartite='False'
        a_node='MESH:D012559'
        c_node='PR:000012942'
        gold_b='b'
        cutoff_year='1993'
    fi

    if [ $setup_experiment = True ]
    then
        #Setup experiment
        echo "Setting up experiment."

        if [ $embedding_creation_method = 'node2vec' ]
        then
          tegf="test_adj_mat_${embeddingsshortname}.edgelist"
        elif [ $embedding_creation_method = 'line' ]
        then
          tegf="test_adj_mat_${embeddingsshortname}.line"
        fi

        python 'create_splits.py' -f "${datapath}/edges_with_scores.csv"  -ci ${ci} -cl ${cl} \
          -a ${a_node} -c ${c_node} -cy ${cutoff_year} -ts '200000' -tf "train_${embeddingsshortname}_${lbd_method}.tsv" \
          -df "devel_${embeddingsshortname}_${lbd_method}.tsv" -tef "test_${embeddingsshortname}_${lbd_method}.tsv" \
          -vf "vertices_${embeddingsshortname}_${lbd_method}.txt" \
          -tegf "${tegf}" --B-filename "b_${embeddingsshortname}_${lbd_method}.txt" \
          --C-filename "Cs_${embeddingsshortname}_${lbd_method}.txt" --lbd_method "${lbd_method}"
    fi

    #create representations
    echo "\n----------------------------------------------------------"
    echo "----------------------------------------------------------"
    if [ $create_representations = True ]
    then
        #Create representations
        echo "Embedding creation method: ${embedding_creation_method}."

        if [ $embedding_creation_method = 'node2vec' ]
        then
            #Efficient node2vec
            ./node2vec_cpp/node2vec -i:"test_adj_mat_${embeddingsshortname}.edgelist" \
              -o:"test_${embeddingsshortname}.embeddings" -l:80 -d:100 -p:2 -q:4 -k:10 -e:2 -v -w
        elif [ $embedding_creation_method = 'line' ]
        then
            LINE/linux/line -train "test_adj_mat_${embeddingsshortname}.line" \
              -output "test_${embeddingsshortname}-order1.embeddings" -size 50 -order 1 \
              -samples 1000 -threads 10 #Halve so the combined vector can have the desired dimension

            LINE/linux/line -train "test_adj_mat_${embeddingsshortname}.line" \
              -output "test_${embeddingsshortname}-order2.embeddings" -size 50 -order 2 \
              -samples 1000 -threads 10 #Halve so the combined vector can have the desired dimension

            #Concatenate and normalise as recomended in paper
            echo "Combining LINE vectors..."
            python line_combine.py -i1 "test_${embeddingsshortname}-order1.embeddings" \
              -i2 "test_${embeddingsshortname}-order2.embeddings" -o "test_modified_${embeddingsshortname}.embeddings"
        else
            echo "ERROR: UNKNOWN NODE CREATION METHOD!"
        fi

        if [ $embedding_creation_method != 'line' ]
        then
            #Create the modified embeddings to change from node indices to node names
            echo "Creating modified embeddings."
            python 'create_modified_embeddings.py' -f "test_${embeddingsshortname}.embeddings" \
              -o "test_modified_${embeddingsshortname}.embeddings"
        fi

        #Convert the embeddings to .bin format
        echo "Converting embeddings to .bin format."
        python 'wvlib/convert.py' -i sdv "test_modified_${embeddingsshortname}.embeddings" "test_modified_${embeddingsshortname}.embeddings.bin"
    fi

    #Do LBD
    if [ $do_lbd = True ]
    then
      for experiment_run in 1 2 3 4 5
      do
        echo "\n----------------------------------------------------------"
        echo "Experiments run: ${experiment_run}"
        echo "----------------------------------------------------------"

        for combination_method in hadamard average concatenate weighted_l2 weighted_l1
        do
          echo "\n----------------------------------------------------------"
          echo "Training model with ${combination_method}."

          #Train model
          echo "LBD method is: ${lbd_method}."
          if [ $lbd_method = 'closed_discovery_with_aggregators' ]
          then
              python 'Models/MLP/pytorch/neural_link_scorer_2node_input.py' --train_data "train_${embeddingsshortname}_${lbd_method}.tsv" \
                --test_data "test_${embeddingsshortname}_${lbd_method}.tsv" \
                --test_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --train_epochs 150 \
                --combination_method ${combination_method} --a_node ${a_node} --c_node ${c_node} \
                --goldb_node ${gold_b} --experiment_name "${dataset}_${combination_method}_${experiment_run}" \
                --lbd_type "closed_discovery"
          elif [ $lbd_method = 'closed_discovery_without_aggregators' ]
          then
              python 'Models/MLP/pytorch/neural_predictor_3node_input.py' --train_data "train_${embeddingsshortname}_${lbd_method}.tsv" \
              --test_data "test_${embeddingsshortname}_${lbd_method}.tsv" \
              --test_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --train_epochs 150 \
              --combination_method ${combination_method}  \
              --a_node ${a_node} --c_node ${c_node} --goldb_node ${gold_b} --experiment_name "${dataset}_${combination_method}_${experiment_run}" \
              --lbd_type "closed_discovery_abc"
          elif [ $lbd_method = 'open_discovery_with_aggregators_and_accumulators' ]
          then
            python 'Models/MLP/pytorch/neural_link_scorer_2node_input.py' --train_data "train_${embeddingsshortname}_${lbd_method}.tsv" \
              --test_data "test_${embeddingsshortname}_${lbd_method}.tsv" \
              --test_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --train_epochs 150 \
              --combination_method ${combination_method}  \
              --a_node ${a_node} --c_node ${c_node} --goldb_node ${gold_b} --experiment_name "${dataset}_${combination_method}_${experiment_run}" \
              --c_list "Cs_${embeddingsshortname}_${lbd_method}.txt" --lbd_type "open_discovery"
          elif [ $lbd_method = 'open_discovery_without_aggregators_and_accumulators' ]
          then
              python 'Models/CNN/pytorch/cnn_predict_score.py' --train_data "train_${embeddingsshortname}_${lbd_method}.tsv" \
              --test_data "test_${embeddingsshortname}_${lbd_method}.tsv" --test_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" \
              --train_epochs 5 --combination_method ${combination_method}  \
              --a_node ${a_node} --c_node ${c_node} --goldb_node ${gold_b} --experiment_name "${dataset}_${combination_method}_${experiment_run}" \
              --b_list "b_${embeddingsshortname}.txt" --c_list "Cs_${embeddingsshortname}_${lbd_method}.txt" --lbd_type "open_discovery_ac"
          fi
        done
      done
    fi
done
