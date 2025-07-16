import { useContext, useEffect, useRef, useState } from "react"
import AccountForm from "../../components/AccountForm/AccountForm"
import { DataContext } from "../../context/DataContext"
import { Email, Person, LockOutline } from '@mui/icons-material'
import FormPage from "../../components/FormPage/FormPage"
import AuthenticationMessage from "../../components/AuthenticationMessage/AuthenticationMessage"
import GoogleButton from "../../components/GoogleButton/GoogleButton"
import { crud } from "../../api/crud"
import { useNavigate } from "react-router-dom"

const Register = () => {
    // Gets global data from the context
    const { access, setLoading } = useContext(DataContext)



    // Checks if the user is already authenticated
    useEffect(() => {
        if(access) useNavigate()('/dashboard')
    }, [access])



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
        setLoading(true)

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

        setLoading(false)
    }



    return (
        <>
            <AuthenticationMessage
                open={modal}
                onClose={() => setModal(false)}
                title="Thanks for signing up!"
                message="We've sent an activation link to your email."
                email={emailRef.current ? emailRef.current.value : null}
                text="Once you're done activating your account, you can"
            />
            <FormPage>
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
                    buttonLabel="Create my account"
                    oauth={{
                        component: (
                            <GoogleButton
                                setError={setError}
                                text="signup_with"
                            />
                        )
                    }}
                />
            </FormPage>
        </>
    )
}

export default Register