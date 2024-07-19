def LatexParameters(dc):
    parameters = f"""
            $input\_current$ & ${dc["input_current"]}$ \\\\ \hline  
            $input\_model$ & ${dc["input_model"]}$ \\\\ \hline 
            $input\_size$ & ${dc["input_size"]}$ \\\\ \hline 
            $input\_time$ & ${dc["input_time"]}$ \\\\ \hline 
            $rest\_time$ & ${dc["rest_time"]}$ \\\\ \hline 
            $duration\_time$ & ${dc["duration_time"]}$ \\\\ \hline 
            $iteration$ & ${dc["iteration"]}$ \\\\ \hline 
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
        \caption{Experiment Parameters related to figure number | part 1}
    \end{table}
    ------------"""
    )
    print(res)
