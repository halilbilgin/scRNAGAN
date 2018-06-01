diff_gene_expression <- function(train_raw, train_labels, generated_raw, generated_labels) {
    library(ROTS)
    fdrResults <- rep(NA, ncol(generated_labels))

    for(cell_type in 1:ncol(generated_labels)){

        input = rbind(train_raw[train_labels[, cell_type] == 1, ],
                    generated_raw[generated_labels[, cell_type]==1,])

        groups = c(rep(1, length(which(train_labels[, cell_type]==1))),
                 rep(2, length(which(generated_labels[, cell_type]==1))))

        results = ROTS(data = t(input), groups = groups, B = 500,
                                K = ncol(train_raw)*0.2, seed = 1234)
        summary_results = summary(results, fdr = 0.05)

        fdrResults[[cell_type]] = (ncol(input) - dim(summary_results)[1]) / ncol(input)

        gc()
    }

    return(fdrResults)
}