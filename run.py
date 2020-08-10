from HousingModel import HousingModelClass
updpar = dict()
updpar["Nbeta"] = 1
updpar["Na"] = 50
model = HousingModelClass(name="baseline",solmethod="None",**updpar)
model.test()
