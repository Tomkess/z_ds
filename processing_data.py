import pyspark.sql.functions as F

def process_data_frame(data_frame):
    """
    Process data frame - do feature selection and engineering
    :param data_frame: data set to be processed
    :return: processed data frame
    """

    # creates expression to map categorrical feature by given mapping
    def get_expression_for_mapping(mapping, column):
        return "case {} else 'other' end".format(" ".join([" ".join(["when {} = '{}' then '{}'".format(column, v, k)
                                                                     for v in val]) for k, val in mapping.items()]))

    # creates expression to map categorical feature by given mapping to boolean
    def get_expression_for_boolean_mapping(values_to_be_set_true, column):
        return "case {} else False end".format(" ".join(["when {} = '{}' then True".format(column, v)
                                                         for v in values_to_be_set_true]))

    # define mappings
    education_mapping = {'with_degree': [' Doctorate degree(PhD EdD)', ' Associates degree-academic program',
                                         ' Bachelors degree(BA AB BS)', ' Prof school degree (MD DDS DVM LLB JD)',
                                         ' Associates degree-occup /vocational',
                                         ' Masters degree(MA MS MEng MEd MSW MBA)'],
                         'with_high_school': [' High school graduate'],
                         'with_college': [' Some college but no degree']}
    class_of_worker_mapping = [' Private']
    marital_status_mapping = [' Married-A F spouse present', ' Married-civilian spouse present']
    sex_mapping = [' Male']
    tax_filer_status_mapping = {'nonfiler': [' Nonfiler'],
                                ' joint_under_65': [' Joint both under 65'],
                                'single': [' Single']}
    detailed_household_summary_in_household_mapping = {'child': [' Child 18 or older', ' Child under 18 never married',
                                                                 ' Child under 18 ever married'],
                                                       'householder': [' Householder'],
                                                       'spouse_of_householder': [' Spouse of householder']}
    family_members_under_18_mapping = [' Not in universe']

    # - processing on data frame
    return data_frame.select('age', 'num_persons_worked_for_employer',
                             'race', 'member_of_a_labor_union', 'capital_gains', 'capital_losses',
                             'citizenship', 'weeks_worked_in_year', 'full_or_part_time_employment_stat',
                             F.expr("case when class == ' - 50000.' then 'No' else 'Yes' end").alias('has_over_50k'),
                             F.expr('case when veterans_benefits == 2 then True else False end').alias(
                                 'has_veterans_benefits'),
                             F.expr('case when weeks_worked_in_year > 0 then True else False end').alias(
                                 'worked_in_year'),
                             F.expr(get_expression_for_mapping(tax_filer_status_mapping, 'tax_filer_status')).alias(
                                 'tax_filer_status'),
                             F.expr(get_expression_for_mapping(education_mapping, 'education')).alias('education'),
                             F.expr(get_expression_for_mapping(detailed_household_summary_in_household_mapping,
                                                               'detailed_household_summary_in_household')).alias(
                                 'household_summary'),
                             F.expr(
                                 get_expression_for_boolean_mapping(class_of_worker_mapping, 'class_of_worker')).alias(
                                 'is_private_worker_class'),
                             F.expr(get_expression_for_boolean_mapping(marital_status_mapping, 'marital_status')).alias(
                                 'is_married'),
                             F.expr(get_expression_for_boolean_mapping(sex_mapping, 'sex')).alias('is_male'),
                             F.expr(get_expression_for_boolean_mapping(family_members_under_18_mapping,
                                                                       'family_members_under_18')).alias(
                                 'is_family_members_under_18_out_of_universe'))
