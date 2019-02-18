#!/bin/sh

setup_experiment=False
create_representations=False
do_lp=True

#Various settings
#options: 'open_discovery_with_aggregators_and_accumulators' 'open_discovery_without_aggregators_and_accumulators'
lbd_method='open_discovery_with_aggregators_and_accumulators'

dev_eval=True
test_eval=True

echo "\n----------------------------------------------------------"
echo "----------------------------------------------------------"
echo "----------------------------------------------------------"
echo "----------------------------------------------------------"

for dataset in biogrid
do
    echo "\n----------------------------------------------------------"
    echo "----------------------------------------------------------"
    echo "----------------------------------------------------------"
    echo "Using dataset ${dataset}."

    ci='0:1:2:9'
    cl=':START_ID,:END_ID,year:int,metric_jaccard:float[]'

    if [ "$dataset" = "biogrid" ]
    then
        datapath='data/biogrid'
        embeddingsshortname='biogrid_ts'
        cutoff_year='2016'
    fi

    if [ $setup_experiment = True ]
    then
        #Setup experiment
        python 'experiment_setup.py' -if "${datapath}/edges.csv"  -ci ${ci} -cl ${cl} \
        -cy ${cutoff_year} -idf "${datapath}/dev.tsv" -itf "${datapath}/test.tsv" \
        -dif "${datapath}/discoverable_edges_biogrid.json" -ts '200000' \
        -tf "train_${embeddingsshortname}_${lbd_method}.tsv" -df "devel_${embeddingsshortname}_${lbd_method}.tsv" \
        -tef "test_${embeddingsshortname}_${lbd_method}.tsv" -vf "vertices_${embeddingsshortname}_${lbd_method}.txt" \
        -tegf "test_adj_mat_${embeddingsshortname}.edgelist" --B-filename "b_${embeddingsshortname}_${lbd_method}.txt" \
        --C-filename "Cs_${embeddingsshortname}_${lbd_method}.txt" --lbd_method "${lbd_method}"
    fi

    #create representations
    for method in node2vec #node2vec #line
    do
        echo "\n----------------------------------------------------------"
        echo "----------------------------------------------------------"
        if [ $create_representations = True ]
        then
           echo "Method: ${method}."

           if [ $method = 'node2vec' ]
           then
                echo "Creating node representations with node2vec"
                node2vec_cpp/node2vec -i:"test_adj_mat_${embeddingsshortname}.edgelist" -o:"test_${embeddingsshortname}.embeddings" -l:80 -d:100 -p:2 -q:2 -k:10 -e:2 -v -w
            elif [ $method = 'line' ]
            then
                echo "Creating node representations with LINE"
                LINE/linux/line -train "test_adj_mat_${embeddingsshortname}.line" -output "test_${embeddingsshortname}-order1.embeddings" -size 50 -order 1 -samples 1000 -threads 10 #HALVE SO THAT COMBINED CAN HAVE DESIRED DIM
                LINE/linux/line -train "test_adj_mat_${embeddingsshortname}.line" -output "test_${embeddingsshortname}-order2.embeddings" -size 50 -order 2 -samples 1000 -threads 10 #HALVE SO THAT COMBINED CAN HAVE DESIRED DIM
                #Concatenate and normalise as recomended in paper
                python line_combine.py -i1 "test_${embeddingsshortname}-order1.embeddings" -i2 "test_${embeddingsshortname}-order2.embeddings" -o "test_modified_${embeddingsshortname}.embeddings"
            else
                echo "UNKNOWN NODE CREATION METHOD!"
            fi

            if [ $method != 'line' ]
            then
                #Create the modified embeddings to change from node indices to node names
                echo "Creating modified embeddings."
                python 'create_modified_embeddings.py' -f "test_${embeddingsshortname}.embeddings" -o  "test_modified_${embeddingsshortname}.embeddings" -vf "vertices_${embeddingsshortname}_${lbd_method}.txt"
            fi

            #Convert the embeddings to .bin format
            echo "Converting embeddings to .bin format."
            python 'wvlib/convert.py' -i sdv "test_modified_${embeddingsshortname}.embeddings" "test_modified_${embeddingsshortname}.embeddings.bin"
        fi

        if [ $do_lp = True ]
        then
          for experiment_run in 1 2 3 4 5
          do
            devel_filename="${datapath}/dev.tsv"
            devel_unformed_filename="${datapath}/reachable_dev_top_1000.json"
            devel_data="devel_${embeddingsshortname}_${lbd_method}.tsv"
            eval_filename="${datapath}/test.tsv"
            unformed_filename="${datapath}/reachable_test_top_1000.json"
            test_data="test_${embeddingsshortname}_${lbd_method}.tsv"

            for combination_method in concatenate hadamard concatenate weighted_l2 average weighted_l1
            do
                echo "\n----------------------------------------------------------"
                echo "Training convnet with ${combination_method}."
                #Train model
                echo "LBD method is: ${lbd_method}."
                if [ $lbd_method = 'open_discovery_with_aggregators_and_accumulators' ]
                then
                  python 'Models/MLP/pytorch/neural_link_scorer_2node_input_no_cases.py' --train_data "train_${embeddingsshortname}_${lbd_method}.tsv" \
                    --devel_data "${devel_data}" --test_data "test_${embeddingsshortname}_${lbd_method}.tsv" \
                    --test_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --eval_filename "${eval_filename}" --unformed_filename "${unformed_filename}"  \
                    --devel_filename "${devel_filename}" --devel_unformed_filename "${devel_unformed_filename}" --train_epochs 1 \
                    --combination_method ${combination_method} --experiment_name "${dataset}_${combination_method}_${experiment_run}" \
                     --lbd_type "open_discovery"
                elif [ $lbd_method = 'open_discovery_without_aggregators_and_accumulators' ]
                then
                  python 'Models/CNN/pytorch/cnn_no_cases_predict_score.py' --train_data "train_${embeddingsshortname}_${lbd_method}.tsv" --devel_data "${devel_data}" --test_data "${test_data}" \
                  --test_embeddings_data "test_modified_${embeddingsshortname}.embeddings.bin" --eval_filename "${eval_filename}" --unformed_filename "${unformed_filename}"  \
                  --devel_filename "${devel_filename}" --devel_unformed_filename "${devel_unformed_filename}" \
                  --train_epochs 20 --combination_method ${combination_method} --method ${method}  --a_node ${a_node} --c_node ${c_node} --goldb_node ${gold_b} \
                  --experiment_name "${combination_method}_${experiment_run}" --b_list "b_${embeddingsshortname}.txt" --c_list "Cs_${embeddingsshortname}.txt" --lbd_type "open_discovery_ac" \
                  -if "${datapath}/edges.csv"  -ci ${ci} -cl ${cl} -cy ${cutoff_year}
                fi
            done
          done
        fi
      done
done
