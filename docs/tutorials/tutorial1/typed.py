import theano
import theano.tensor as tt
import numpy as np


def method(age, sex, educ, race, income, married, N):

    @theano.compile.ops.as_op(itypes=[tt.lvector, tt.lvector, tt.lvector,
        tt.lvector, tt.dvector, tt.lvector, tt.lscalar], otypes=[tt.dscalar])
    def dp_program(age, sex, educ, race, income, married, N):
        import opendp.smartnoise.core as sn
        import pandas as pd
        temp_file = 'temp.csv'
        var_names = ['age', 'sex', 'educ', 'race', 'income', 'married']
        data = {'age': age, 'sex': sex, 'educ': educ, 'race': race,
            'income': income, 'married': married}
        df = pd.DataFrame(data, columns=var_names)
        df.to_csv(temp_file)
        with sn.Analysis() as analysis:
            data = sn.Dataset(path=temp_file, column_names=var_names)
            age_mean = sn.dp_mean(data=sn.to_float(data['income']),
                privacy_usage={'epsilon': 0.1}, data_lower=0.0, data_upper=
                200.0, data_rows=N)
        analysis.release()
        return np.float64(age_mean.value)
    return dp_program(age, sex, educ, race, income, married, N)
