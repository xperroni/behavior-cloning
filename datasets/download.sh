#!/bin/bash

function download {
    wget -nc -O $1 $2
    tar -jvxf $1
}


download track1-train-01.tar.bz2 https://www.researchgate.net/profile/Helio_Perroni_Filho/publication/311769799_Behavioral_Learning_training_data_0103/links/5859f50208ae3852d2559f79
download track1-train-02.tar.bz2 https://www.researchgate.net/profile/Helio_Perroni_Filho/publication/311774469_Behavioral_Learning_training_data_0203/links/585a47c708ae3852d256f0fd
download track1-train-03.tar.bz2 https://www.researchgate.net/profile/Helio_Perroni_Filho/publication/311774954_Behavioral_Learning_training_data_0303/links/585a53c608ae64cb3d4a999d

download track1-validation.tar.bz2 https://www.researchgate.net/profile/Helio_Perroni_Filho/publication/311766863_Behavioral_Learning_validation_data/links/585970a108ae64cb3d494091
