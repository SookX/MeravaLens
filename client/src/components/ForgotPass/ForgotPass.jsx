import { useContext, useRef, useState } from "react"
import FormPage from "../FormPage/FormPage"
import AccountForm from "../AccountForm/AccountForm"
import { DataContext } from "../../context/DataContext"
import AuthenticationMessage from "../AuthenticationMessage/AuthenticationMessage"

const ForgotPass = () => {
    // Gets global data from the context
    const { crud } = useContext(DataContext)
    


    // Holds the values for the form
    const emailRef = useRef()
    const [error, setError] = useState(false)
    const [modal, setModal] = useState(false)

    const inputs = [
        {
            type: "email",
            label: "Email",
            ref: emailRef
        }
    ]



    // Sends a request to the backend for a password reset link
    const handleForgotPassword = async () => {
        const response = await crud({
            url: "/users/forgot-password/",
            method: "post",
            body: {
                email: emailRef.current.value
            }
        })
        
        console.log(response)

        if(response.status == 200) setModal(true)
        else setError(response.response.data.error)
    }



    return (
        <>
            <AuthenticationMessage
                open={modal}
                onClose={() => setModal(false)}
                title="Your new password is on its way!"
                message="We've sent a password reset link to:"
                email={emailRef.current ? emailRef.current.value : null}
                text="Once you've reset your password, you can"
            />
            <FormPage>
                <AccountForm
                    title="Forgotten Password"
                    text="Don't worry! Enter your account's email and we'll immediately send you a password reset link."
                    error={error}
                    inputs={inputs}
                    handleSubmit={handleForgotPassword}
                    link={{
                        text: "Found your password?",
                        label: "Log in",
                        link: "/login"
                    }}
                    buttonLabel="Get my link"
                />
            </FormPage>
        </>
    )
}

export default ForgotPass