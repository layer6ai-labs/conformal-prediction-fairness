<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

# Fairness of Conformal Prediction Dataset (FCPD)
We are releasing the experimental data collected and analyzed in the work ["Conformal Prediction Sets Can Cause Disparate Impact"](https://openreview.net/forum?id=fZK6AQXlUU), accepted as a Spotlight paper to ICLR 2025.

For each of the three experiments done on FACET, BiosBias, and RAVDESS, we provide a set of three csv files.

## Dataset and Prediction Data

The dataset file (e.g. `facet.csv`) contains the dataset and prediction-related information which was shown to human participants. The columns are `[idx, prompt, label_text, label, original_label, group, group_text, top1, topk_set, avgk_set, conformal_marginal_set, conformal_conditional_set, corr_ans_text, topk_text, avgk_text, conformal_marginal_text, conformal_conditional_text]`.
Explanations of columns:\
`idx`: A unique id for joining.\
`prompt`: The text datapoint or file location of an image datapoint.\
`label`: The ground truth classification label, after remapping.\
`original_label`: The ground truth classification label before remapping, which matches the original dataset.\
`group`: An integer representing which group the datapoint belongs to, used for fairness analysis.\
`top1`: The top-1 class predicted by the model.\
`avgk_set`: The set of classes predicted by the Average-k method.\
`conformal_marginal_set`: The set of classes predicted by the Average-k method.\
`conformal_conditional_set`: The set of classes predicted by the Average-k method.\

## Individual Response Data
The responses file `individual_results.csv` contains the human response data. Each row represents a single datapoint shown to a single participant. The columns are `[experiment, participant_id, prompt_idx, response, label, label_text, group, group_text, top1, avgk_set, conformal_marginal_set, conformal_conditional_set]`.
Explanations of columns:\
`experiment`: Describes the treatment that was applied, one of `control`, `avgk`, `marginal`, `conditional`.\
`participant_id`: A unique id for each of the 600 participants in the study.\
`prompt_idx`: A unique id describing which datapoint was shown. Can be joined on `idx` from the dataset file described above.\
`response`: The class predicted by the participant.\
The remaining data matches the columns in the dataset file described above, by `prompt_idx`/`idx`.

## Participant Data
The results file `results.csv` contains data aggregated over the individual responses, along with demographic data at the participant level. Each row represents a single participant in the study. The columns are `[participant_id, experiment, num_correct, time_taken, Age, Sex, Ethnicity]`. 
Explanations of columns:\
`participant_id`: A unique id for each of the 600 participants in the study. Matches the ids from the individual responses.\
`experiment`: Describes the treatment that was applied, one of `control`, `avgk`, `marginal`, `conditional`.\
`num_correct`: The number of correct reponses given by the participant, out of 50 trials.\
`time_taken`: The number of seconds the participant used in responding to the 50 trials.\
`Age`: The age of the participant, provided by the Prolific platform.\
`Sex`: The binary gender of the participant, provided by the Prolific platform. One of `Female`, `Male`, or `Prefer not to say`.\
`Ethnicity`: A simplified version of the ethnicity of the participant, provided by the Prolific platform.\