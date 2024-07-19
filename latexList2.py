def LatexParameters(dc):
    parameters = f"""
            $input\_current$ & ${dc["input_current"]}$ \\\\ \hline  
            $output\_current$ & ${dc["output_current"]}$ \\\\ \hline 
            $input\_model$ & ${dc["input_model"]}$ \\\\ \hline 
            $output\_model$ & ${dc["output_model"]}$ \\\\ \hline 
            $input\_size$ & ${dc["input_size"]}$ \\\\ \hline 
            $output\_size$ & ${dc["output_size"]}$ \\\\ \hline 
            $input\_time$ & ${dc["input_time"]}$ \\\\ \hline 
            $rest\_time$ & ${dc["rest_time"]}$ \\\\ \hline 
            $duration\_time$ & ${dc["duration_time"]}$ \\\\ \hline 
            $iteration$ & ${dc["iteration"]}$ \\\\ \hline 
            $positive\_learning\_rate$ & ${dc["positive_learning_rate"]}$ \\\\ \hline 
            $negative\_learning\_rate$ & ${dc["negative_learning_rate"]}$ \\\\ \hline 
            $input\_encoding\_method$ & ${dc["input_encoding_method"]}$ \\\\ \hline 
            $synapse\_model$ & ${dc["synapse_model"]}$ \\\\ \hline 
            $synapse\_J_0$ & ${dc["synapse_j0"]}$ \\\\ \hline 
            $synapse\_weight\_deviation$ & ${dc["deviation"]}$ \\\\ \hline 
            $synapse\_current\_tau$ & ${dc["synapse_current_tau"]}$ \\\\ \hline 
            $training\_rule$ & ${dc["training_rule"]}$ \\\\ \hline 
            $tau\_input\_trace$ & ${dc["tau_input_trace"]}$ \\\\ \hline 
            $tau\_output\_trace$ & ${dc["tau_output_trace"]}$ \\\\ \hline 
            $normalization$ & ${dc["normalization"]}$ \\\\ \hline 
            """
    res = (
        """-----------
    \\begin{table}[htbp]
        \centering
        \\begin{tabular}{||l|l||}
            \hline
            \\textbf{Parameter}  & \\textbf{Value} \\\\ \hline \hline \hline
    """
        + parameters
        + """
        \end{tabular}
        \caption{Experiment Parameters related to \\textbf{figure number } | \\textbf{part 2}}
    \end{table}
    ------------"""
    )
    print(res)
