import { useContext, useEffect, useRef, useState } from "react"
import AccountForm from "../../components/AccountForm/AccountForm"
import { DataContext } from "../../context/DataContext"
import FormPage from "../../components/FormPage/FormPage"
import GoogleButton from "../../components/GoogleButton/GoogleButton"
import { crud } from "../../api/crud"
import { useNavigate } from "react-router-dom"

const Login = () => {
    // Gets global data from the context
    const { access, setAccess, setRefresh, setLoading } = useContext(DataContext)



    // Checks if the user is already authenticated
    useEffect(() => {
        if(access) useNavigate()('/dashboard')
    }, [access])



    // Holds the values for the form
    const emailRef = useRef()
    const passwordRef = useRef()
    const rememberMeRef = useRef()
    const [error, setError] = useState(null)

    const inputs = [
        {
            type: "email",
            label: "Email",
            ref: emailRef
        },
        {
            type: "password",
            label: "Password",
            ref: passwordRef
        }
    ]



    // Makes a login request to the backend
    const handleSubmit = async () => {
        setLoading(true)

        const response = await crud({
            url: "/users/login/",
            method: "post",
            body: {
                email: emailRef.current.value,
                password: passwordRef.current.value,
            }
        })

        if(response.status == 200) {
            if(rememberMeRef.current.checked) {
                localStorage.setItem('access', response.data.token.access)
                localStorage.setItem('refresh', response.data.token.refresh)
            } else {
                sessionStorage.setItem('access', response.data.token.access)
                sessionStorage.setItem('refresh', response.data.token.refresh)
            }
            setAccess(response.data.token.access)
            setRefresh(response.data.token.refresh)
            useNavigate()('/dashboard')
        }
        else setError(response.response.data.error)

        console.log(response)

        setLoading(false)
    }



    return (
        <FormPage>
            <AccountForm
                title="Welcome back!"
                text="Enter your credentials and get back to the exciting world of satellites."
                error={error}
                inputs={inputs}
                forgotPassword={true}
                rememberMeRef={rememberMeRef}
                handleSubmit={handleSubmit}
                link={{
                    link: "/register",
                    text: "You don't have an account?",
                    label: "Sign up"
                }}
                buttonLabel="Log in to my account"
                oauth={{
                    component: (
                        <GoogleButton
                            setError={setError}
                            text="signin_with"
                        />
                    )
                }}
            />
        </FormPage>
    )
}

export default Login