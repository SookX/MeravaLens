import { useContext, useRef, useState } from "react"
import AccountForm from "../../components/AccountForm/AccountForm"
import FormPage from "../../components/FormPage/FormPage"
import { DataContext } from "../../context/DataContext"
import { crud } from "../../api/crud"
import { useNavigate } from "react-router-dom"

const ChangePassword = () => {
    // Gets global data from the context
    const { setLoading } = useContext(DataContext)



    // Navigates users to another page
    const navigate = useNavigate()



    // Holds the values for the form
    const oldPasswordRef = useRef()
    const newPasswordRef = useRef()
    const [error, setError] = useState(null)

    const inputs = [
        {
            type: "password",
            label: "Old Password",
            ref: oldPasswordRef
        },
        {
            type: "password",
            label: "New Password",
            ref: newPasswordRef
        },
    ]



    // Makes a change password request to the backend
    const handleChangePass = async () => {
        setLoading(true)

        const response = await crud({
            url: "/users/me/",
            method: "put",
            body: {
                old_password: oldPasswordRef.current.value,
                new_password: newPasswordRef.current.value
            }
        })

        if(response.status == 200) navigate('/dashboard')
        else setError(response.response.data.message)

        setLoading(false)
    }



    return (
        <FormPage>
            <AccountForm
                title="Change password"
                text="To change your password, type your old and new password in the fields below."
                error={error}
                inputs={inputs}
                handleSubmit={handleChangePass}
                buttonLabel="Change my password"
            />
        </FormPage>
    )
}

export default ChangePassword