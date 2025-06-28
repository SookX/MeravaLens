import { useContext, useRef, useState } from "react"
import AccountForm from "../../components/AccountForm/AccountForm"
import { DataContext } from "../../context/DataContext"
import { Email, Person, LockOutline } from '@mui/icons-material'
import ActivateMessage from "./components/ActivateMessage/ActivateMessage"

const Register = () => {
    // Gets global data from the context
    const { crud } = useContext(DataContext)



    // Holds the values for the form
    const emailRef = useRef()
    const usernameRef = useRef()
    const passwordRef = useRef()
    const confirmPasswordRef = useRef()
    const [error, setError] = useState(null)
    const [modal, setModal] = useState(false)

    const inputs = [
        {
            type: "email",
            label: "Email",
            ref: emailRef
        },
        {
            type: "text",
            label: "Username",
            ref: usernameRef
        },
        {
            type: "password",
            label: "Password",
            ref: passwordRef
        },
        {
            type: "password",
            label: "Confirm Password",
            ref: confirmPasswordRef
        },
    ]


    // Makes a register request to the backend
    const handleSubmit = async () => {
        const response = await crud({
            url: "/users/register/",
            method: "post",
            body: {
                email: emailRef.current.value,
                username: usernameRef.current.value,
                password: passwordRef.current.value,
                confirmPassword: confirmPasswordRef.current.value
            }
        })

        if(response.status == 201) setModal(true)
        else setError(response.response.data.error)

        console.log(response)
    }



    return (
        <>
            <ActivateMessage open={modal} onClose={() => setModal(false)} /*email={emailRef.current.value}*/ email="velchev061@gmail.com" />
            <AccountForm
                title="Make an account."
                text="Make a Merava Lens account to get the latest satellite analysis."
                error={error}
                inputs={inputs}
                handleSubmit={handleSubmit}
                link={{
                    link: "/login",
                    text: "Alredy have an account?",
                    label: "Log in"
                }}
            />
        </>
    )
}

export default Register