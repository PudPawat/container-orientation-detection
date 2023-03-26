from PyQt5 import QtWidgets,uic
class set_config_result_print():
    def set_step():
        source = call.input_folder_box.text()
        resize_true = call.true_resize.isChecked()
        resize_false = call.false_resize.isChecked()
        process_list = []
        for process_name in range(1, 16):
            exec(f'step_{process_name} = call.step{process_name}_box.currentText()')
            exec(f'if "-- Select Step" not in step_{process_name}: process_list.append(step_{process_name})')
        # configpath = call.save_config_path_box.text()
        # confignameformat = call.save_config_name_box.text()


        print("source:", source)
        # print("source1:", source1)
        if(resize_true == "False"):
            resize_status = resize_false
        else:
            resize_status = resize_true
        print("resize:", resize_status)
        print("process:", process_list)
        # print("config_path:", configpath)
        # print("config_name_format:", confignameformat)

    def set_save_location():
        configpath = call.save_config_path_box.text()
        # confignameformat = call.save_config_name_box.text()
        print("config_path:", configpath)
        # print("config_name_format:", confignameformat)

# class get_save_location():

app=QtWidgets.QApplication([])
call=uic.loadUi("set_config.ui")

# call.confirm.clicked.connect(set_config_result_print.set_step)
call.confirm.clicked.connect(set_config_result_print.set_save_location)

call.show()
app.exec()