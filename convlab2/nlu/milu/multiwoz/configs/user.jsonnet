{
  "dataset_reader": {
    "type": "milu",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      },
    },
    "context_size": 3,
    "agent": "user" 
  },
  "train_data_path": "../../../data/multiwoz/train.json.zip",
  "validation_data_path": "../../../data/multiwoz/val.json.zip",
  "test_data_path": "../../../data/multiwoz/test.json.zip",
  "model": {
    "type": "milu",
    "label_encoding": "BIO",
    "dropout": 0.3,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": 50,
            "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.50d.txt.gz",
            "trainable": true
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "embedding_dim": 16
            },
            "encoder": {
            "type": "cnn",
            "embedding_dim": 16,
            "num_filters": 128,
            "ngram_filter_sizes": [3],
            "conv_layer_activation": "relu"
            }
        }
      }
    },
    "encoder": {
      "type": "lstm",
      "input_size": 178,
      "hidden_size": 200,
      "num_layers": 1,
      "dropout": 0.5,
      "bidirectional": true
    },
    "intent_encoder": {
      "type": "lstm",
      "input_size": 400,
      "hidden_size": 200,
      "num_layers": 1,
      "dropout": 0.5,
      "bidirectional": true
    },
    "attention": {
      "type": "bilinear",
      "vector_dim": 400,
      "matrix_dim": 400
    },    
    "context_for_intent": true,
    "context_for_tag": false,
    "attention_for_intent": false,
    "attention_for_tag": false,
    "regularizer": [
      [
        "scalar_parameters",
        {
          "type": "l2",
          "alpha": 0.1
        }
      ]
    ]
  },
  "iterator": {
    "type": "basic",
    "batch_size": 64
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": 0.001
    },
    "validation_metric": "+f1-measure",
    "num_serialized_models_to_keep": 3,
    "num_epochs": 40,
    "grad_norm": 5.0,
    "patience": 75,
    "cuda_device": 0 
  },
  "evaluate_on_test": true
}
