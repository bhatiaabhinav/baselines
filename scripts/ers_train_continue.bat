taskkill /f /im "ERSEnv*"
"%GYM_PYTHON%" -m baselines.a2c.run_general --env=%1 --num_cpu=8 --saved_model="%OPENAI_LOGDIR%\%1\%2\model" %3 %4 %5 %6 %7 %8 %9