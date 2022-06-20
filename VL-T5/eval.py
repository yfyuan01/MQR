import pickle
import language_evaluation
import json
with open('result_t5p.pkl','rb') as f:
    result_dict = pickle.load(f)
with open('result.pkl','rb') as f:
    result_ref = pickle.load(f)
# evaluator = language_evaluation.CocoEvaluator()
evaluator = language_evaluation.RougeEvaluator(num_parallel_calls=5)
rouge_results = evaluator.run_evaluation(result_dict['predictions'],result_dict['targets'])
print('Rouge results:')
print(rouge_results)
n_c = 0
p_c = 0
n = 0
p = 0

for i in range(len(result_dict['predictions'])):
    if result_dict['predictions'][i] == result_dict['targets'][i]:
        if result_dict['targets'][i] == result_ref['questions'][i]:
            n_c += 1
        else:
            p_c += 1
    if result_dict['targets'][i] == result_ref['questions'][i]:
        n += 1
    else:
        p += 1
        
p_EM = p_c/float(p)
n_EM = n_c/float(n)
print(f'Positive EM:{p_EM}')
print(f'Negative EM:{n_EM}')

            